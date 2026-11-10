"""Route source-channel U views using cross-receiver branch evidence."""
from contextlib import contextmanager

import torch

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from .recovery import branch_certificate, installed as labeled_scope


def capture_pair(student, clean, changed):
    modes = [(m, m.training) for m in student.modules()]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(clean.device) if clean.is_cuda else None
    try:
        student.eval()
        core, sequence, reference = _actual_paths(student)
        with torch.no_grad():
            with capture_branches(core, sequence, reference) as raw:
                student(clean)
            with capture_branches(core, sequence, reference) as altered:
                student(changed)
    finally:
        for module, mode in modes:
            module.training = mode
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, clean.device)
    return raw, altered


def select_views(offered, certified, mode, seed):
    if mode == 'all':
        return offered
    if mode == 'branch':
        return offered & certified
    if mode == 'random':
        choices = offered.nonzero(as_tuple=True)[0]
        count = int((offered & certified).sum())
        generator = torch.Generator(device='cpu').manual_seed(seed)
        order = torch.randperm(len(choices), generator=generator)[:count].to(choices.device)
        selected = torch.zeros_like(offered)
        selected[choices[order]] = True
        return selected
    raise ValueError('Unknown practical U routing mode')


class PracticalGate(GuardedReceiverExtrapolation):
    def __init__(self, c):
        super().__init__(c)
        self.labeled = None
        self.as_calls = self.as_offered = self.as_used = self.as_certified = 0

    def validate_unlabeled_batch(self, batch):
        super().validate_unlabeled_batch(batch)
        visible = batch[3]['tx_label_visible']
        if visible.dtype != torch.bool or visible.shape != batch[1].shape or bool(visible.any()):
            raise ValueError('Only hidden source U can use simulated channel views')

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        if self.labeled is None:
            raise ValueError('Current source labeled anchors required')
        indices = physical.nonzero(as_tuple=True)[0]
        selected_meta = {k: v[indices] if torch.is_tensor(v) and v.ndim and len(v) == len(x) else v
                         for k, v in metadata.items()}
        simulated = student._cd_practical(x[indices], selected_meta, epoch, step)
        raw_l, labels, l_rx = self.labeled
        clean = torch.cat((raw_l, x[indices]))
        changed = torch.cat((raw_l, simulated))
        raw, altered = capture_pair(student, clean, changed)
        _, certified = branch_certificate(raw, altered, labels, l_rx, metadata['rx_i'][indices],
                                          pseudo[indices], self.c['num_classes'], 4)
        offered = accepted[indices]
        recipe = self.c['practical_gate']
        mode = recipe.get('mode', 'branch' if recipe.get('branch_gate') else 'all')
        selected = select_views(offered, certified, mode, self.c['model_seed'] * 100_000 + step)
        strength = recipe.get('strength', 1.)
        mixed = ordinary.clone()
        mixed[indices] = x[indices]
        view = simulated[selected]
        if strength != 1.:
            view = x[indices[selected]] + strength * (view - x[indices[selected]])
        mixed[indices[selected]] = view
        self.as_calls += 1
        self.as_offered += int(offered.sum())
        self.as_certified += int((offered & certified).sum())
        self.as_used += int(selected.sum())
        return mixed

    def state_dict(self):
        state = super().state_dict()
        recipe = self.c['practical_gate']
        mode = recipe.get('mode', 'branch' if recipe.get('branch_gate') else 'all')
        if (self.labeled is not None or self.as_calls != self.calls
                or not 0 <= self.as_certified <= self.as_offered <= self.seen // 2
                or self.as_used != (self.as_offered if mode == 'all' else self.as_certified)):
            raise ValueError('Simulated U views must match the registered source gate')
        # Native counters assume all physical slots were raw when hard_pl_no_uas is set.
        state['raw_As_seen'] -= self.as_used
        state['physical_seen'] += self.as_used
        state['physical_student_views'] += 2 * self.as_used
        state['U_practical_gate'] = dict(calls=self.as_calls, offered=self.as_offered,
            certified=self.as_certified, used=self.as_used, recipe=recipe)
        return state


@contextmanager
def installed(c, holder):
    with labeled_scope(c, holder, objective_factory=PracticalGate):
        yield
