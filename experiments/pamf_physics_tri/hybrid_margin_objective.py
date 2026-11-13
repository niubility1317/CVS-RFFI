"""Replace the extra fusion contrast with a capped classification margin."""
import torch
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_ssl_research.hard_pl_joint_objective import capture_joint
from .hybrid_margin import capped_margin_loss, source_margin_cap
from .hybrid_structure import HybridStructure


class HybridMargin(HybridStructure):
    def __init__(self, c, probe=None):
        super().__init__(c, probe)
        self.margin_cap = None
        self.margin_violations = self.margin_accepted = 0

    def prepare_margin_cap(self, student):
        """Freeze the source-L median once, then reuse it for the E30 segment."""
        path = Path(self.c['output_root']).parent / 'margin_cap.json'
        if self.c.get('epoch_offset') == 30 and not self.c.get('hard_pl_smoke'):
            self.margin_cap = read(path)['margin_cap']
            return
        from experiments.cric_dura_comparison.f_data import SourceDataset
        source = SourceDataset(self.c['source_cache'], 'L_s', read(self.c['source_contract']))
        device = next(student.parameters()).device
        devices = [device.index] if device.type == 'cuda' else []
        modes = [(m, m.training) for m in student.modules()]
        features, labels = [], []
        try:
            student.eval()
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for x, y, domain, _ in torch.utils.data.DataLoader(source, batch_size=128, shuffle=False):
                    with capture_joint(student.id_backbone.encoder) as joint:
                        student.forward_identity_only(x.to(device), domain_labels=domain.to(device))
                    features.append(joint[0].cpu())
                    labels.append(y)
        finally:
            for module, training in modes:
                module.training = training
        weights = student.id_backbone.encoder.core.id_backbone.cls_head.weight.detach().cpu()
        cap, correct = source_margin_cap(torch.cat(features), torch.cat(labels), weights)
        self.margin_cap = float(cap)
        if not self.c.get('hard_pl_smoke'):
            write(path, dict(margin_cap=self.margin_cap, rule='median correctly classified source-L margin',
                             source_samples=len(source), correct_samples=correct,
                             source_checkpoint='same-row E20; first labeled forward already updated BN buffers',
                             U_truth_reads=0, target_reads=0))

    def loss(self, student, device):
        if self.margin_cap is None:
            self.prepare_margin_cap(student)
        try:
            return super().loss(student, device)
        finally:
            self.raw_aux_view = None

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        self.raw_aux_view = x
        return super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)

    def additional_fusion_loss(self, views, selected):
        student = self.aux_student
        pseudo, accepted = self.hybrid_labels
        modes = [(m, m.training) for m in student.modules()]
        devices = [self.aux_view.device.index] if self.aux_view.is_cuda else []
        try:
            student.eval()
            with torch.random.fork_rng(devices=devices):
                with torch.no_grad(), capture_joint(student.id_backbone.encoder) as raw:
                    student.forward_identity_only(self.raw_aux_view, domain_labels=self.aux_domain)
                with capture_joint(student.id_backbone.encoder) as altered:
                    logits = student.forward_identity_only(self.aux_view, domain_labels=self.aux_domain)['tx_logits']
        finally:
            for module, training in modes:
                module.training = training
        weights = student.id_backbone.encoder.core.id_backbone.cls_head.weight
        margin, detail = capped_margin_loss(raw[0], altered[0], weights, pseudo, accepted, self.margin_cap)
        if self.c.get('hard_pl_smoke'):
            named = [(n, p) for n, p in student.named_parameters() if p.requires_grad and
                     ('.sequence_pool.' in n or '.response_projection.' in n)]
            gradients = torch.autograd.grad(margin, [p for _, p in named], retain_graph=True, allow_unused=True)
            norms = {'signature': 0., 'reference': 0.}
            for (name, _), gradient in zip(named, gradients):
                if gradient is not None:
                    group = 'signature' if '.sequence_pool.' in name else 'reference'
                    norms[group] += float(gradient.detach().double().square().sum())
            self.hybrid_smoke_gradients = {'margin': {key: value ** .5 for key, value in norms.items()}}
        weighted = self.c['hybrid_auxiliary_weight'] * margin
        recipe = self.c['hard_pl_joint_contrast']
        self.aux_loss = self.settings['weight'] * recipe['weight'] / recipe['CE_normalizer'] * weighted
        self.hybrid_calls += 1
        self.hybrid_anchors += int(accepted.sum())
        self.margin_accepted += int(accepted.sum())
        self.margin_violations += int(detail['violating'].sum())
        self.metrics.update(HPL_hybrid_margin=margin.detach(), HPL_aux_added=self.aux_loss.detach(),
                            HPL_margin_violating=int(detail['violating'].sum()))
        self.trace_batch.update(aux_prediction_1=logits.detach().argmax(1).cpu().tolist(),
            aux_prediction_2=[None] * len(pseudo), aux_added=float(self.aux_loss.detach()),
            margin_reference=detail['reference'].cpu().tolist(),
            margin_augmented=detail['augmented_margin'].cpu().tolist(),
            margin_violating=detail['violating'].cpu().tolist(), margin_violation=detail['amount'].cpu().tolist())
        if not self.c.get('hard_pl_smoke'):
            self.write_trace(self.trace_batch)
        return weighted

    def state_dict(self):
        state = super().state_dict()
        state['U_margin'] = dict(cap=self.margin_cap, accepted=self.margin_accepted,
                                 violations=self.margin_violations, class_weights_detached=True,
                                 raw_reference_detached=True, input='isolated original and physical views')
        return state
