"""Separate primary U routing from optional fusion-only physics supervision."""
import torch
from experiments.pamf_ssl_research.hard_pl_joint_objective import capture_joint
from .hybrid_tracked import HybridTracked
from .hybrid_confidence import HybridConfidence
from .packet_contrast import packet_contrast_loss
from .practical_gate import view_strength
from .branch_guard import GROUPS, parameter_group
from experiments.pamf_ssl_research.u_gradprotect_objective import projected_gradients
from pathlib import Path


class HybridStructure(HybridTracked):
    def __init__(self, c, probe=None):
        super().__init__(c)
        self.update_probe = probe

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        if self.c['structure_arm'] == 'A':
            # No simulated view in the zero-auxiliary, ordinary/raw baseline.
            indices = accepted.nonzero(as_tuple=True)[0]
            self.record_practical_selection(indices, len(x))
            n = len(indices)
            self.as_calls += 1; self.as_offered += n; self.as_used += n
            self.as_certified += n; self.as_active_certified += n
            self.accepted_original_physical_half += int((accepted & physical).sum())
            self.accepted_classes.update(pseudo[accepted].detach().cpu().tolist())
            self.trace_batch = dict(epoch=epoch, step=step, packet=metadata['base_index'].detach().cpu().tolist(),
                rx=metadata['rx_i'].detach().cpu().tolist(), day=metadata['day_i'].detach().cpu().tolist(),
                confidence=self.trace_confidence.cpu().tolist(), pseudo=pseudo.detach().cpu().tolist(),
                accepted=accepted.detach().cpu().tolist(),physical_strength=0.,main_view_routed=0,aux_physical_nonzero=0)
            mixed = ordinary.clone(); mixed[physical] = x[physical]
            return mixed
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        strength=view_strength(self.c['practical_gate'],epoch)
        self.trace_batch.update(physical_strength=strength,
            main_view_routed=int(accepted.sum()) if self.c['structure_arm'] in ('B','D') else 0,
            aux_physical_nonzero=int(accepted.sum()) if self.c['structure_arm'] in ('C','D') and strength>0 else 0)
        if self.c['structure_arm'] in ('B', 'D'):
            self.as_physical_used += int((accepted & physical).sum())
            return self.aux_view
        return mixed

    def additional_fusion_loss(self, views, selected):
        pseudo, accepted = self.hybrid_labels
        student = self.aux_student
        if self.c['structure_arm'] in ('A', 'B'):
            zero = views[0].sum() * 0.
            self.aux_loss = zero
            self.hybrid_calls += 1
            self.metrics.update(HPL_hybrid_fusion_contrast=zero.detach(), HPL_aux_added=zero.detach())
            n = len(pseudo)
            self.trace_batch.update(aux_prediction_1=[None]*n, aux_prediction_2=[None]*n, aux_added=0.)
            if not self.c.get('hard_pl_smoke'):
                self.write_trace(self.trace_batch)
            if self.c.get('hard_pl_smoke'):
                self.hybrid_smoke_gradients = {}
            return zero

        bn = [(m, m.training) for m in student.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        devices = [self.aux_view.device.index] if self.aux_view.is_cuda else []
        altered, predictions = [], []
        try:
            for module, _ in bn: module.eval()
            with torch.random.fork_rng(devices=devices):
                for _ in range(2):
                    with capture_joint(student.id_backbone.encoder) as feature:
                        logits = student.forward_identity_only(self.aux_view, domain_labels=self.aux_domain)['tx_logits']
                    altered.append(feature[0]); predictions.append(logits.detach().argmax(1))
        finally:
            for module, training in bn: module.train(training)
        one_class = bool(selected.any()) and pseudo[accepted].unique().numel() == 1
        def pair(a, b):
            if one_class:
                return ((1 - torch.nn.functional.cosine_similarity(a.detach(), b, dim=1, eps=1e-8))*selected).mean()
            return packet_contrast_loss(a, b, pseudo, accepted, selected, temperature=.2)[0]
        fusion = torch.stack([pair(views[i], altered[i]) for i in range(2)]).mean()
        if one_class: self.fallback_calls += 1
        if self.c.get('hard_pl_smoke'):
            named = [(n,p) for n,p in student.named_parameters() if p.requires_grad and
                     ('.sequence_pool.' in n or '.response_projection.' in n)]
            grads = torch.autograd.grad(fusion,[p for _,p in named],retain_graph=True,allow_unused=True)
            norms={'signature':0.,'reference':0.}
            for (name,_),grad in zip(named,grads):
                if grad is not None: norms['signature' if '.sequence_pool.' in name else 'reference'] += float(grad.detach().double().square().sum())
            self.hybrid_smoke_gradients={'fusion':{k:v**.5 for k,v in norms.items()}}
        weighted = self.c['hybrid_auxiliary_weight'] * fusion
        scale = self.settings['weight'] * self.c['hard_pl_joint_contrast']['weight'] / self.c['hard_pl_joint_contrast']['CE_normalizer']
        self.aux_loss = scale * weighted
        self.hybrid_calls += 1; self.hybrid_anchors += 2 * int(selected.sum())
        self.metrics.update(HPL_hybrid_fusion_contrast=fusion.detach(), HPL_aux_added=self.aux_loss.detach())
        self.trace_batch.update(aux_prediction_1=predictions[0].cpu().tolist(),aux_prediction_2=predictions[1].cpu().tolist(),
                                aux_added=float(self.aux_loss.detach()))
        if not self.c.get('hard_pl_smoke'): self.write_trace(self.trace_batch)
        return weighted

    def state_dict(self):
        state = super().state_dict()
        enabled = self.c['structure_arm'] in ('C','D')
        state['U_fused_physics'].update(extra_student_forwards=2 if enabled else 0, auxiliary_enabled=enabled,
                                        selected=state['U_fused_physics']['selected'] if enabled else 0)
        state['U_confidence_only']['used'] = state['accepted'] if enabled else 0
        state['U_tracking']['auxiliary_enabled'] = enabled
        state['U_tracking']['same_state_adamw_observations'] = self.update_probe.calls if self.update_probe else 0
        return state

    def guide(self,labeled,student):
        if self.c['structure_arm'] in ('C','D') and self.calls % 50 == 0 and not self.c.get('hard_pl_smoke'):
            named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
            params=[p for _,p in named]
            gl=torch.autograd.grad(labeled,params,retain_graph=True,allow_unused=True)
            total=torch.autograd.grad(self.pending_gradient_loss,params,retain_graph=True,allow_unused=True)
            aux=torch.autograd.grad(self.aux_loss,params,retain_graph=True,allow_unused=True)
            base=tuple(None if a is None and b is None else
                       (torch.zeros_like(p) if a is None else a)-(torch.zeros_like(p) if b is None else b)
                       for (_,p),a,b in zip(named,total,aux))
            ratio=self.c['hard_pl_gradient_guard']['U_to_L_norm_cap']
            protected,details=projected_gradients(named,gl,base,max_u_ratio=ratio,group_fn=parameter_group,groups=GROUPS)
            self.update_probe.capture(student,named,protected,self.calls,Path(self.c['output_root'])/'U_update_probe.jsonl.gz')
        return super().guide(labeled,student)
