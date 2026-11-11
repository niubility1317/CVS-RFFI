"""Identity consistency of the deployed fusion under certified source channels."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from experiments.pamf_ssl_research.hard_pl_joint_objective import capture_joint
from .single_rx import SingleRxDay
from .practical_gate import installed as practical_scope


def paired_fusion_loss(raw, altered, selected):
    if raw.shape != altered.shape or raw.ndim != 2 or selected.dtype != torch.bool or selected.shape != raw.shape[:1]:
        raise ValueError('Aligned fused features and exact source-channel selection required')
    return ((1. - F.cosine_similarity(raw.detach(), altered, dim=1, eps=1e-8)) * selected).mean()


class FusedPhysics(SingleRxDay):
    def __init__(self, c):
        super().__init__(c)
        self.physical_selection = None
        self.fusion_views = None
        self.fusion_calls = self.fusion_selected = 0

    def record_practical_selection(self, indices, batch_size):
        if self.physical_selection is not None:
            raise ValueError('One physical selection per source-U loss required')
        self.physical_selection = torch.zeros(batch_size, device=indices.device, dtype=torch.bool)
        self.physical_selection[indices] = True

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self._rxe_forward_index >= 4:
            return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        with capture_joint(student.id_backbone.encoder) as captured:
            result = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        self.fusion_views.append(captured[0])
        return result

    def loss(self, student, device):
        self.fusion_views = []
        try:
            result = super().loss(student, device)
            if len(self.fusion_views) != 4 or self.physical_selection is None or self.pending_gradient_loss is None:
                raise ValueError('Existing raw and physically routed U fusion views required')
            values = [paired_fusion_loss(self.fusion_views[i], self.fusion_views[i + 2], self.physical_selection)
                      for i in range(2)]
            relation = torch.stack(values).mean()
            if self.c.get('hard_pl_smoke'):
                named = [(name, p) for name, p in student.named_parameters() if p.requires_grad and
                         ('.sequence_pool.' in name or '.response_projection.' in name)]
                gradients = torch.autograd.grad(relation, [p for _, p in named], retain_graph=True, allow_unused=True)
                norms = {'signature': 0., 'reference': 0.}
                for (name, _), gradient in zip(named, gradients):
                    if gradient is not None:
                        if not bool(torch.isfinite(gradient).all()):
                            raise ValueError('Finite fused physics gradients required')
                        group = 'signature' if '.sequence_pool.' in name else 'reference'
                        norms[group] += float(gradient.detach().double().square().sum())
                self.fusion_smoke_gradients = {key: value ** .5 for key, value in norms.items()}
            recipe = self.c['hard_pl_joint_contrast']
            scale = self.settings['weight'] * recipe['weight'] / recipe['CE_normalizer']
            self.pending_gradient_loss = self.pending_gradient_loss + scale * relation
            self.fusion_calls += 1
            self.fusion_selected += int(self.physical_selection.sum())
            self.metrics.update(HPL_fused_physics_loss=relation.detach(), HPL_fused_physics_weight=scale,
                                HPL_fused_physics_selected=self.fusion_selected)
            return result
        finally:
            self.fusion_views = None
            self.physical_selection = None

    def state_dict(self):
        state = super().state_dict()
        if (self.fusion_views is not None or self.physical_selection is not None or self.fusion_calls != self.calls
                or self.fusion_selected != self.as_used):
            raise ValueError('One same-packet physical fusion loss per source-U loss required')
        state['U_fused_physics'] = dict(calls=self.fusion_calls, selected=self.fusion_selected,
            coefficient='original joint contrast coefficient and CE normalizer', raw_features_detached=True,
            extra_student_forwards=0, physical_inputs='existing branch-certified source As')
        if self.c.get('hard_pl_smoke'):
            state['U_fused_physics']['smoke_gradients'] = self.fusion_smoke_gradients
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .fused_physics_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=FusedPhysics):
        yield
