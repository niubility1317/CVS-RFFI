"""Physically probed branch attenuation for unlabeled source examples."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from experiments.pamf_branch_training.objective import _actual_paths
from .practical_gate import capture_pair, installed as practical_scope
from .single_rx import SingleRxDay

PATHS = ('time', 'frequency', 'pa', 'path', 'reference')


def routed_paths(raw, changed, mode, positions, step):
    if raw.keys() != changed.keys() or set(raw) != set(PATHS) or mode not in ('balanced', 'physical', 'complementary'):
        raise ValueError('Five aligned deployed source-U paths required')
    n = len(positions)
    if any(raw[key].shape != changed[key].shape or raw[key].ndim != 2 or len(raw[key]) != n for key in PATHS):
        raise ValueError('Aligned source-U physical probe features required')
    with torch.no_grad():
        similarity = torch.stack([F.cosine_similarity(raw[key], changed[key], dim=1, eps=1e-8)
                                  for key in PATHS], dim=1)
        if not bool(torch.isfinite(similarity).all()):
            raise ValueError('Finite physical probe similarities required')
        order = similarity.argsort(dim=1)
        if mode == 'balanced':
            first = (positions.long() + step) % len(PATHS)
            second = first
        else:
            first = order[:, 0]
            second = order[:, 1] if mode == 'complementary' else first
    return first, second


@contextmanager
def attenuated_branches(student, assignment, selected, factor):
    if assignment.shape != selected.shape or selected.dtype != torch.bool or not 0 < factor < 1:
        raise ValueError('One attenuation assignment per selected physical U packet required')
    core, sequence, reference = _actual_paths(student)
    modules = ((core.t_proj,), (core.f_proj, core.freq_stats_proj),
               (core.pa_proj, core.pa_stats_proj), (sequence,), (reference,))
    handles = []
    try:
        for index, group in enumerate(modules):
            mask = selected & (assignment == index)
            for module in group:
                def hook(_module, _inputs, output, mask=mask):
                    if not isinstance(output, torch.Tensor) or output.ndim != 2 or len(output) != len(mask):
                        raise ValueError('Expected one deployed vector per source-U packet')
                    return output * torch.where(mask[:, None], factor, 1.)
                handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


class PhysicalBranchDropout(SingleRxDay):
    def __init__(self, c):
        super().__init__(c)
        self.route_selected = self.route_assignments = None
        self.route_calls = self.route_selected_count = self.route_forwards = self.route_probes = 0
        self.route_counts = {name: 0 for name in PATHS}

    def record_practical_selection(self, indices, batch_size):
        if self.route_selected is not None:
            raise ValueError('Exactly one physical selection per source-U batch')
        self.route_selected = torch.zeros(batch_size, device=indices.device, dtype=torch.bool)
        self.route_selected[indices] = True

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        if self.route_selected is None or self.route_assignments is not None:
            raise ValueError('Certified source-U physical slots must be recorded')
        indices = self.route_selected.nonzero(as_tuple=True)[0]
        assignments = [torch.zeros(len(x), device=x.device, dtype=torch.long) for _ in range(2)]
        if len(indices):
            raw, changed = capture_pair(student, x[indices], mixed[indices])
            first, second = routed_paths(raw, changed, self.c['physical_branch_dropout']['mode'], indices, step)
            assignments[0][indices], assignments[1][indices] = first, second
            self.route_probes += 2
        self.route_assignments = assignments
        self.route_selected_count += len(indices)
        return mixed

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        index = self._rxe_forward_index
        if index in (2, 3):
            if self.route_assignments is None or self.route_selected is None:
                raise ValueError('Physical branch assignments must precede physical U forward')
            assignment = self.route_assignments[index - 2]
            with attenuated_branches(student, assignment, self.route_selected,
                                     self.c['physical_branch_dropout']['factor']):
                result = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
            self.route_forwards += 1
            for i, name in enumerate(PATHS):
                self.route_counts[name] += int((self.route_selected & (assignment == i)).sum())
            return result
        return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)

    def loss(self, student, device):
        try:
            result = super().loss(student, device)
            self.route_calls += 1
            return result
        finally:
            self.route_selected = self.route_assignments = None

    def state_dict(self):
        state = super().state_dict()
        if (self.route_selected is not None or self.route_assignments is not None or self.route_calls != self.calls
                or self.route_selected_count != self.as_used or self.route_forwards != 2 * self.calls
                or sum(self.route_counts.values()) != 2 * self.route_selected_count
                or not 0 <= self.route_probes <= 2 * self.calls):
            raise ValueError('Physical branch routing must cover both existing U physical views')
        state['U_physical_branch_dropout'] = dict(recipe=self.c['physical_branch_dropout'], calls=self.route_calls,
            selected=self.route_selected_count, routed_by_branch=self.route_counts,
            student_physical_forwards=self.route_forwards, extra_source_u_eval_forwards=self.route_probes,
            raw_mask_unchanged=True, U_truth_read=False, target_access=False)
        return state


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .physical_branch_dropout_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=PhysicalBranchDropout):
        yield
