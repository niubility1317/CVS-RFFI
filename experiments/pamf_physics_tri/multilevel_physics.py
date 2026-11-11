"""Same-packet source-channel consistency in branch and prediction spaces."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import capture_branches
from .fused_physics import paired_fusion_loss
from .single_rx import SingleRxDay
from .practical_gate import installed as practical_scope


def paired_prediction_loss(raw_a, raw_b, changed_a, changed_b, selected):
    if (raw_a.shape != raw_b.shape or raw_a.shape != changed_a.shape or raw_a.shape != changed_b.shape
            or raw_a.ndim != 2 or selected.dtype != torch.bool or selected.shape != raw_a.shape[:1]):
        raise ValueError('Aligned same-packet U logits and boolean physical selection required')
    target = (.5 * (raw_a.softmax(1) + raw_b.softmax(1))).detach()
    losses = [F.kl_div(value.log_softmax(1), target, reduction='none').sum(1)
              for value in (changed_a, changed_b)]
    return (.5 * (losses[0] + losses[1]) * selected).mean()


def paired_branch_loss(raw_a, raw_b, changed_a, changed_b, selected):
    if not raw_a or raw_a.keys() != raw_b.keys() or raw_a.keys() != changed_a.keys() or raw_a.keys() != changed_b.keys():
        raise ValueError('Identical deployed branch paths required')
    return torch.stack([.5 * (paired_fusion_loss(raw_a[k], changed_a[k], selected)
                              + paired_fusion_loss(raw_b[k], changed_b[k], selected)) for k in raw_a]).mean()


class MultiLevelPhysics(SingleRxDay):
    def __init__(self, c):
        super().__init__(c)
        self.physical_selection = self.branch_views = self.prediction_views = None
        self.level_calls = self.level_selected = 0

    def record_practical_selection(self, indices, batch_size):
        if self.physical_selection is not None:
            raise ValueError('One source-channel selection per U loss required')
        self.physical_selection = torch.zeros(batch_size, device=indices.device, dtype=torch.bool)
        self.physical_selection[indices] = True

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        capture = self._rxe_forward_index < 4
        if capture and self.c['multilevel_physics']['branch']:
            core, sequence, reference = _actual_paths(student)
            with capture_branches(core, sequence, reference) as packet:
                logits = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
            self.branch_views.append(packet)
        else:
            logits = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        if capture and self.c['multilevel_physics']['soft']:
            self.prediction_views.append(logits)
        return logits

    def loss(self, student, device):
        self.branch_views, self.prediction_views = [], []
        try:
            result = super().loss(student, device)
            if self.physical_selection is None or self.pending_gradient_loss is None:
                raise ValueError('Existing certified physical U slots required')
            extra = {}
            if self.c['multilevel_physics']['branch']:
                if len(self.branch_views) != 4:
                    raise ValueError('Two raw and two perturbed deployed branch views required')
                extra['branch'] = paired_branch_loss(*self.branch_views, self.physical_selection)
            if self.c['multilevel_physics']['soft']:
                if len(self.prediction_views) != 4:
                    raise ValueError('Two raw and two perturbed existing logits required')
                extra['soft'] = paired_prediction_loss(*self.prediction_views, self.physical_selection)
            if self.c.get('hard_pl_smoke'):
                self.level_smoke_gradients = {}
                named = [(name, p) for name, p in student.named_parameters() if p.requires_grad and
                         ('.sequence_pool.' in name or '.response_projection.' in name)]
                for kind, loss in extra.items():
                    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
                    norms = {'signature': 0., 'reference': 0.}
                    for (name, _), gradient in zip(named, gradients):
                        if gradient is not None:
                            if not bool(torch.isfinite(gradient).all()):
                                raise ValueError('Finite physical consistency gradients required')
                            group = 'signature' if '.sequence_pool.' in name else 'reference'
                            norms[group] += float(gradient.detach().double().square().sum())
                    self.level_smoke_gradients[kind] = {key: value ** .5 for key, value in norms.items()}
            recipe = self.c['hard_pl_joint_contrast']
            normalizer = recipe['CE_normalizer']
            if 'branch' in extra:
                self.pending_gradient_loss = self.pending_gradient_loss + self.settings['weight'] * recipe['weight'] / normalizer * extra['branch']
            if 'soft' in extra:
                self.pending_gradient_loss = self.pending_gradient_loss + self.settings['weight'] * .25 / normalizer * extra['soft']
            self.level_calls += 1
            self.level_selected += int(self.physical_selection.sum())
            self.metrics.update({f'HPL_physical_{kind}_loss': loss.detach() for kind, loss in extra.items()})
            return result
        finally:
            self.physical_selection = self.branch_views = self.prediction_views = None

    def state_dict(self):
        state = super().state_dict()
        if (self.physical_selection is not None or self.branch_views is not None or self.prediction_views is not None
                or self.level_calls != self.calls or self.level_selected != self.as_used):
            raise ValueError('Source-channel consistency count or context differs')
        state['U_multilevel_physics'] = dict(recipe=self.c['multilevel_physics'], calls=self.level_calls,
            selected=self.level_selected, raw_features_and_teacher_detached=True, extra_student_forwards=0)
        if self.c.get('hard_pl_smoke'):
            state['U_multilevel_physics']['smoke_gradients'] = self.level_smoke_gradients
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .multilevel_physics_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=MultiLevelPhysics):
        yield
