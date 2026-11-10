"""Training-only physical direction measurements and source branch losses."""
from contextlib import contextmanager
from types import MethodType
import math
import torch

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_ssl_research.hard_pl_objective import HardPLObjective
from .geometry import bounded_echo, physical_branch_loss, tangent


def select_indices(mask, limit, seed):
    if mask.ndim != 1 or mask.dtype != torch.bool or limit <= 0:
        raise ValueError('Boolean source selection and positive limit required')
    choices = torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten()
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    order = torch.randperm(len(choices), generator=generator)[:limit]
    return choices[order].to(mask.device)


def select_u_candidates(accepted, limit, seed):
    """Match physical exposure even when the two students accept different U."""
    if accepted.ndim != 1 or accepted.dtype != torch.bool:
        raise ValueError('Boolean U acceptance mask required')
    return select_indices(torch.ones_like(accepted), limit, seed)


def echo_parameters(count, device, dtype, seed):
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    amplitude = .04 + .06 * torch.rand(count, generator=generator)
    phase = 2 * math.pi * torch.rand(count, generator=generator)
    coefficients = torch.stack((amplitude * phase.cos(), amplitude * phase.sin()), dim=1).to(device=device, dtype=dtype)
    delays = torch.randint(1, 4, (count,), generator=generator).to(device=device)
    return coefficients, delays


def branch_views(student, inputs, seed):
    """Capture deployed branches before and after a bounded echo."""
    if inputs.ndim != 3 or inputs.shape[1:] != (2, 256) or not len(inputs):
        raise ValueError('Nonempty 256-sample source IQ required')
    coefficients, delays = echo_parameters(len(inputs), inputs.device, inputs.dtype, seed)
    transformed = bounded_echo(inputs, coefficients, delays)
    modes = [(m, m.training) for m in student.modules()]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
    try:
        student.eval()
        core, sequence, reference = _actual_paths(student)
        with torch.no_grad():
            with capture_branches(core, sequence, reference) as raw:
                student(inputs)
            with capture_branches(core, sequence, reference) as changed:
                student(transformed)
    finally:
        for module, mode in modes:
            module.training = mode
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, inputs.device)
    return raw, changed


def physical_probes(student, inputs, seed):
    raw, changed = branch_views(student, inputs, seed)
    directions, norms = {}, {}
    for name in raw:
        directions[name], norms[name] = tangent(raw[name], changed[name])
    return raw, directions, norms


def physical_directions(student, inputs, seed):
    """Same-input eval/eval differences; restore modes and global RNG exactly."""
    _, directions, norms = physical_probes(student, inputs, seed)
    return directions, norms


def source_relation(packet, directions, labels, metadata, selected, c, accepted=None, *, protect_identity=False):
    device = labels.device
    rx = metadata['rx_i'].to(device).long().index_select(0, selected)
    days = metadata['day_i'].to(device).long().index_select(0, selected)
    ids = metadata['branch_record_id'].to(device).long().index_select(0, selected)
    if set(rx.tolist()) - set(c['source_rxs']) or set(days.tolist()) - set(c['source_days']):
        raise ValueError('Physical relations require declared source RX/day')
    chosen = labels.index_select(0, selected)
    selected_accepted = (torch.ones_like(chosen, dtype=torch.bool) if accepted is None
                         else accepted.index_select(0, selected))
    branch = {k: v.index_select(0, selected) for k, v in packet.items()}
    if protect_identity:
        if accepted is not None:
            raise ValueError('Identity protection requires visible source labels')
        from .geometry import protect_labeled_identity
        directions = {k: protect_labeled_identity(v, directions[k], chosen) for k, v in branch.items()}
    return physical_branch_loss(branch, directions, chosen, rx, days, ids,
           selected_accepted, beta=c['physical_relation']['beta'],
           single_receiver=len(c['source_rxs']) == 1, reference_scale=c.get('reference_scale', 1.))


def install_labeled(student, config, *, protect_identity=False):
    original_context = student._phase1_labeled_feature_context
    original_hook = student._source_labeled_feature_hook
    student._physical_pending = None
    student._physical_L_calls = student._physical_L_samples = 0

    @contextmanager
    def labeled_context(self):
        if self._physical_pending is not None:
            raise RuntimeError('Unconsumed physical L branch packet')
        core, sequence, reference = _actual_paths(self)
        with original_context():
            with capture_branches(core, sequence, reference) as packet:
                yield
        self._physical_pending = packet

    def hook(self, **kwargs):
        packet, self._physical_pending = self._physical_pending, None
        if packet is None or self._cd_pending is None:
            raise RuntimeError('Actual ordinary L branch/input missing')
        # The model's first L forward uses the native ordinary view, which can
        # differ from labeled_inputs (the unaugmented raw batch).
        ordinary = self._cd_pending['x']
        value, metrics = original_hook(**kwargs)  # removes its active feature hooks
        labels = kwargs['labels']
        step = self._feature_l_steps + config.get('step_offset', 0)
        selected = select_indices(torch.ones_like(labels, dtype=torch.bool), 64, 202611101 + 2 * step)
        directions, norms = physical_directions(self, ordinary.index_select(0, selected), 202611103 + 2 * step)
        branch, counts = source_relation(packet, directions, labels, kwargs['extra'][1], selected, config,
                                         protect_identity=protect_identity)
        self._physical_L_calls += 1
        self._physical_L_samples += len(selected)
        metrics.update(physical_L_loss=branch.detach(), physical_L_calls=self._physical_L_calls,
                       physical_L_samples=self._physical_L_samples, physical_L_valid_anchors=counts['valid_anchors'],
                       physical_L_distance_correction=counts['mean_distance_correction'],
                       **{'physical_L_direction_' + k: v.mean() for k, v in norms.items()})
        return value + config['physical_relation']['l_weight'] * branch, metrics

    hook._accepts_labeled_inputs = True
    student._phase1_labeled_feature_context = MethodType(labeled_context, student)
    student._source_labeled_feature_hook = MethodType(hook, student)
    return student


class PhysicalPL(HardPLObjective):
    def __init__(self, c):
        super().__init__(c)
        if self.settings != dict(method='HPL2', threshold=.95, weight=1.):
            raise ValueError('Exact ordinary PL base required')
        self._physical_U = None
        self.physical_calls = self.physical_samples = self.physical_accepted = 0

    def validate_unlabeled_batch(self, batch):
        super().validate_unlabeled_batch(batch)
        visibility = batch[3].get('tx_label_visible')
        if visibility is None or bool(torch.as_tensor(visibility).any()):
            raise ValueError('Physical U relations require hidden transmitter labels')

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self._physical_U is not None:
            raise RuntimeError('Exactly one U student forward required')
        core, sequence, reference = _actual_paths(student)
        with capture_branches(core, sequence, reference) as packet:
            logits = super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)
        self._physical_U = (packet, x, pseudo, accepted, metadata)
        return logits

    def loss(self, student, device):
        if self._physical_U is not None:
            raise RuntimeError('Unconsumed physical U packet')
        try:
            value = super().loss(student, device)
            packet, inputs, pseudo, accepted, metadata = self._physical_U
            step = student._feature_l_steps + self.c.get('step_offset', 0)
            selected = select_u_candidates(accepted, 128, 202611102 + 2 * step)
            if len(selected):
                directions, norms = physical_directions(student, inputs.index_select(0, selected), 202611104 + 2 * step)
                branch, counts = source_relation(packet, directions, pseudo, metadata, selected, self.c, accepted=accepted)
                details = {'physical_U_direction_' + k: v.mean() for k, v in norms.items()}
            else:
                branch = sum(v.sum() * 0. for v in packet.values())
                counts = dict(valid_anchors=0, mean_distance_correction=branch.detach())
                details = {}
            self.physical_calls += 1
            self.physical_samples += len(selected)
            self.physical_accepted += int(accepted.index_select(0, selected).sum())
            self.metrics.update(physical_U_loss=branch.detach(), physical_U_calls=self.physical_calls,
                                physical_U_samples=self.physical_samples, physical_U_valid_anchors=counts['valid_anchors'],
                                physical_U_accepted=self.physical_accepted,
                                physical_U_distance_correction=counts['mean_distance_correction'], **details)
            return value + self.c['physical_relation']['u_weight'] * branch
        finally:
            self._physical_U = None

    def state_dict(self):
        result = super().state_dict()
        if self._physical_U is not None or self.physical_calls != self.calls:
            raise RuntimeError('Physical U direction must accompany each ordinary PL call')
        result['physical_U'] = dict(calls=self.physical_calls, selected_samples=self.physical_samples,
                                    selected_accepted=self.physical_accepted,
                                    no_grad_forward_packets=2 * self.physical_samples,
                                    target_access=False, U_truth_read=False)
        return result
