"""Preserve source-domain PL knowledge during physical hybrid contrast."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .hybrid_contrast import HybridContrast
from .practical_gate import installed as practical_scope


def distill_pair(first, second, target, selected):
    if (first.shape != second.shape or first.shape != target.shape or first.ndim != 2
            or selected.dtype != torch.bool or selected.shape != first.shape[:1]):
        raise ValueError('Aligned source-U student logits and frozen teacher distribution required')
    values = [F.kl_div(logits.log_softmax(1), target.detach(), reduction='none').sum(1)
              for logits in (first, second)]
    return (.5 * (values[0] + values[1]) * selected).mean()


class TeacherAnchor(HybridContrast):
    def __init__(self, c, teacher):
        super().__init__(c)
        self.teacher = teacher
        self.teacher_target = self.teacher_mask = self.teacher_physical = self.teacher_views = None
        self.teacher_calls = self.teacher_seen = self.teacher_used = self.teacher_physical_used = 0
        self.teacher_disagreements = 0

    def record_practical_selection(self, indices, batch_size):
        super().record_practical_selection(indices, batch_size)
        self.teacher_physical = self.physical_selection.clone()

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        if self.teacher_target is not None or self.teacher.training or any(p.requires_grad for p in self.teacher.parameters()):
            raise ValueError('Exactly one frozen source-only PL teacher per U batch required')
        with torch.no_grad():
            target = self.teacher.classify_features(self.teacher.features(x)).softmax(1)
            confidence, label = target.max(1)
            mask = confidence >= self.settings['threshold']
            disagreement = mask & (label != pseudo)
            if self.c['teacher_anchor']['mode'] == 'agree':
                mask = mask & accepted & (label == pseudo)
        self.teacher_target, self.teacher_mask = target.detach(), mask.detach()
        self.teacher_seen += len(x)
        self.teacher_used += int(mask.sum())
        self.teacher_disagreements += int(disagreement.sum())
        return mixed

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        index = self._rxe_forward_index
        result = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        if index < 4:
            self.teacher_views.append(result)
        return result

    def loss(self, student, device):
        self.teacher_views = []
        try:
            result = super().loss(student, device)
            if (len(self.teacher_views) != 4 or self.teacher_target is None or self.teacher_mask is None
                    or self.teacher_physical is None or self.pending_gradient_loss is None):
                raise ValueError('Frozen teacher and four existing U student views required')
            raw = distill_pair(self.teacher_views[0], self.teacher_views[1], self.teacher_target, self.teacher_mask)
            extra = {'raw': raw}
            if self.c['teacher_anchor']['mode'] == 'teacher_physical':
                mask = self.teacher_mask & self.teacher_physical
                extra['physical'] = distill_pair(self.teacher_views[2], self.teacher_views[3], self.teacher_target, mask)
                self.teacher_physical_used += int(mask.sum())
            if self.c.get('hard_pl_smoke'):
                named = [(name, p) for name, p in student.named_parameters() if p.requires_grad and
                         ('.sequence_pool.' in name or '.response_projection.' in name)]
                self.teacher_smoke_gradients = {}
                for kind, loss in extra.items():
                    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
                    norms = {'signature': 0., 'reference': 0.}
                    for (name, _), gradient in zip(named, gradients):
                        if gradient is not None:
                            if not bool(torch.isfinite(gradient).all()):
                                raise ValueError('Finite teacher-anchor gradients required')
                            group = 'signature' if '.sequence_pool.' in name else 'reference'
                            norms[group] += float(gradient.detach().double().square().sum())
                    self.teacher_smoke_gradients[kind] = {key: value ** .5 for key, value in norms.items()}
            recipe = self.c['hard_pl_joint_contrast']
            coefficient = self.settings['weight'] * recipe['weight'] / recipe['CE_normalizer']
            self.pending_gradient_loss = self.pending_gradient_loss + coefficient * sum(extra.values())
            self.teacher_calls += 1
            self.metrics.update({f'HPL_teacher_{kind}_loss': value.detach() for kind, value in extra.items()})
            return result
        finally:
            self.teacher_target = self.teacher_mask = self.teacher_physical = self.teacher_views = None

    def state_dict(self):
        state = super().state_dict()
        if (any(v is not None for v in (self.teacher_target, self.teacher_mask, self.teacher_physical, self.teacher_views))
                or self.teacher_calls != self.calls or self.teacher_seen != self.seen
                or not 0 <= self.teacher_physical_used <= self.teacher_used <= self.teacher_seen
                or self.teacher.training or any(p.requires_grad for p in self.teacher.parameters())):
            raise ValueError('Frozen same-source teacher execution differs')
        state['U_teacher_anchor'] = dict(recipe=self.c['teacher_anchor'], teacher_source=self.c['teacher_source'],
            calls=self.teacher_calls, seen=self.teacher_seen, raw_used=self.teacher_used,
            physical_used=self.teacher_physical_used, teacher_student_disagreements=self.teacher_disagreements,
            extra_teacher_forwards=self.teacher_calls, extra_student_forwards=0, teacher_frozen=True,
            U_truth_read=False, target_access=False)
        if self.c.get('hard_pl_smoke'):
            state['U_teacher_anchor']['smoke_gradients'] = self.teacher_smoke_gradients
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .teacher_anchor_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_paper_completion.model import load_reader
    from experiments.pamf_paper_completion.plan import make_plan
    from .teacher_anchor_plan import validate
    validate(c)
    row_id = f'source_rx_1-factorial-pl-full-ordered-s{c["model_seed"]}'
    row = next(row for row in make_plan()['rows'] if row['row_id'] == row_id)
    teacher, teacher_cfg = load_reader(c['teacher_source'], read(c['source_contract']), row, 'cuda:0')
    if teacher_cfg['model_seed'] != c['model_seed'] or teacher_cfg['source_rxs'] != c['source_rxs']:
        raise ValueError('Teacher source receiver and model seed must match')
    with practical_scope(c, holder, base_scope=paper_scope,
                         objective_factory=lambda cfg: TeacherAnchor(cfg, teacher)):
        yield
