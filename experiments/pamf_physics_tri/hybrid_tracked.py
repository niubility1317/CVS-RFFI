"""Confidence-complete Hybrid with isolated auxiliary views and packet traces."""
import gzip
import json
import zlib
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_ssl_research.hard_pl_joint_objective import capture_joint
from .branch_guard import GROUPS, parameter_group
from .hybrid_confidence import HybridConfidence
from .packet_contrast import packet_contrast_loss


class HybridTracked(HybridConfidence):
    def __init__(self, c):
        super().__init__(c)
        self.trace_file = None
        self.aux_view = self.trace_batch = self.aux_loss = self.aux_student = self.aux_domain = None
        self.trace_batches = 0
        self.accepted_original_physical_half = 0

    def write_trace(self, batch):
        if self.trace_file is None:
            self.trace_file = gzip.open(Path(self.c['output_root']) / 'U_trace.jsonl.gz', 'wb', compresslevel=1)
        self.trace_file.write((json.dumps(batch, separators=(',', ':')) + '\n').encode('utf8'))
        self.trace_batches += 1
        if self.trace_batches % 50 == 0:
            self.trace_file.flush(zlib.Z_SYNC_FLUSH)

    def accept_pseudo(self, student, x, confidence, pseudo):
        self.trace_confidence = confidence.detach()
        return super().accept_pseudo(student, x, confidence, pseudo)

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        devices = [x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self.aux_view = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        # The original six-forward objective sees the same ordinary/raw partition;
        # the physical view above is consumed only by the extra auxiliary objective.
        physical_accepted = int((accepted & physical).sum())
        self.accepted_original_physical_half += physical_accepted
        self.as_physical_used -= physical_accepted
        mixed = ordinary.clone()
        mixed[physical] = x[physical]
        self.trace_batch = dict(epoch=epoch, step=step,
            packet=metadata['base_index'].detach().cpu().tolist(), rx=metadata['rx_i'].detach().cpu().tolist(),
            day=metadata['day_i'].detach().cpu().tolist(), confidence=self.trace_confidence.cpu().tolist(),
            pseudo=pseudo.detach().cpu().tolist(), accepted=accepted.detach().cpu().tolist())
        return mixed

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self._rxe_forward_index == 0:
            self.aux_domain = domain
        return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)

    def additional_fusion_loss(self, views, selected):
        pseudo, accepted = self.hybrid_labels
        student = self.aux_student
        core, sequence, reference = _actual_paths(student)
        altered, altered_branches, predictions = [], [], []
        bn = [(m, m.training) for m in student.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        devices = [self.aux_view.device.index] if self.aux_view.is_cuda else []
        try:
            for module, _ in bn:
                module.eval()
            with torch.random.fork_rng(devices=devices):
                for _ in range(2):
                    with capture_joint(student.id_backbone.encoder) as feature, capture_branches(core, sequence, reference) as paths:
                        logits = student.forward_identity_only(self.aux_view, domain_labels=self.aux_domain)['tx_logits']
                    altered.append(feature[0]); altered_branches.append(paths)
                    predictions.append(logits.detach().argmax(1))
        finally:
            for module, training in bn:
                module.train(training)

        one_class = bool(selected.any()) and pseudo[accepted].unique().numel() == 1
        if self.c['tracked_aux_mode'] == 'contrast' and one_class:
            self.fallback_calls += 1

        def pair(a, b):
            if self.c['tracked_aux_mode'] == 'positive' or one_class:
                return ((1 - F.cosine_similarity(a.detach(), b, dim=1, eps=1e-8)) * selected).mean()
            return packet_contrast_loss(a, b, pseudo, accepted, selected, temperature=.2)[0]

        fusion = torch.stack([pair(views[i], altered[i]) for i in range(2)]).mean()
        branch = torch.stack([pair(self.hybrid_branches[i][k], altered_branches[i][k])
                              for i in range(2) for k in self.hybrid_branches[i]]).mean()
        raw = fusion + branch
        if self.c.get('hard_pl_smoke'):
            named = [(n, p) for n, p in student.named_parameters() if p.requires_grad
                     and ('.sequence_pool.' in n or '.response_projection.' in n)]
            self.hybrid_smoke_gradients = {}
            for name, component in (('fusion', fusion), ('branch', branch)):
                grads = torch.autograd.grad(component, [p for _, p in named], retain_graph=True, allow_unused=True)
                norms = {'signature': 0., 'reference': 0.}
                for (param, _), grad in zip(named, grads):
                    if grad is not None:
                        key = 'signature' if '.sequence_pool.' in param else 'reference'
                        norms[key] += float(grad.detach().double().square().sum())
                self.hybrid_smoke_gradients[name] = {k: v ** .5 for k, v in norms.items()}
        weighted = self.c['hybrid_auxiliary_weight'] * raw
        scale = self.settings['weight'] * self.c['hard_pl_joint_contrast']['weight'] / self.c['hard_pl_joint_contrast']['CE_normalizer']
        self.aux_loss = scale * weighted
        self.hybrid_calls += 1
        self.hybrid_anchors += 2 * int(selected.sum())
        self.metrics.update(HPL_hybrid_fusion_contrast=fusion.detach(), HPL_hybrid_branch_contrast=branch.detach(),
                            HPL_hybrid_auxiliary_scaled=weighted.detach(), HPL_aux_added=self.aux_loss.detach())
        self.trace_batch['aux_prediction_1'] = predictions[0].cpu().tolist()
        self.trace_batch['aux_prediction_2'] = predictions[1].cpu().tolist()
        self.trace_batch['aux_added'] = float(self.aux_loss.detach())
        if not self.c.get('hard_pl_smoke'):
            self.write_trace(self.trace_batch)
        return weighted

    def loss(self, student, device):
        self.aux_student = student
        try:
            result = super().loss(student, device)
            self.metrics['HPL_complete_U_pre_guard'] = self.pending_gradient_loss.detach()
            return result
        finally:
            self.aux_student = self.aux_view = self.aux_domain = self.trace_batch = None

    def guide(self, labeled, student):
        if self.calls % 50 == 0 and self.aux_loss is not None:
            named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
            params = [p for _, p in named]
            gl = torch.autograd.grad(labeled, params, retain_graph=True, allow_unused=True)
            ga = torch.autograd.grad(self.aux_loss, params, retain_graph=True, allow_unused=True)
            for group in GROUPS:
                pairs = [(a, b) for (n, _), a, b in zip(named, gl, ga) if parameter_group(n) == group and a is not None and b is not None]
                if pairs:
                    dot = sum((a.detach().double() * b.detach().double()).sum() for a, b in pairs)
                    ln = sum(a.detach().double().square().sum() for a, _ in pairs).sqrt()
                    an = sum(b.detach().double().square().sum() for _, b in pairs).sqrt()
                    self.metrics[f'HPL_aux_{group}_dot_L'] = float(dot)
                    self.metrics[f'HPL_aux_{group}_L_norm'] = float(ln)
                    self.metrics[f'HPL_aux_{group}_norm'] = float(an)
        self.aux_loss = None
        return super().guide(labeled, student)

    def state_dict(self):
        state = super().state_dict()
        if self.trace_file is not None:
            self.trace_file.close()
            self.trace_file = None
        state['U_fused_physics'].update(extra_student_forwards=2, physical_inputs='separate auxiliary view')
        state['U_confidence_only']['in_original_physical_half'] = self.accepted_original_physical_half
        state['U_tracking'] = dict(batches=self.trace_batches, packet_key='base_index',
                                   confidence_and_pseudo_per_exposure=True, U_truth_read=False,
                                   path=str(Path(self.c['output_root']) / 'U_trace.jsonl.gz'))
        return state
