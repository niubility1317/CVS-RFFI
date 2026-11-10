"""Preserve PTBR while bounding current-batch reference relation gradients."""
from contextlib import contextmanager
from types import MethodType
import math
import torch

from experiments.pamf_branch_training.objective import _actual_paths
from experiments.pamf_branch_training.relation import capture_branches
from experiments.pamf_physics_branch.objective import (PhysicalPL, physical_directions,
    select_indices, select_u_candidates, source_relation)
from experiments.pamf_ssl_research.hard_pl_objective import HardPLObjective


def gradient_budget(base_loss, reference_loss, parameters, *, floor=.1, eps=1e-12):
    """Detached norm ratio, with no optimizer update or higher-order graph."""
    parameters = tuple(parameters)
    if not parameters or not 0 < floor <= 1 or eps <= 0:
        raise ValueError('Trainable reference parameters and valid budget required')
    base = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
    relation = torch.autograd.grad(reference_loss, parameters, retain_graph=True, allow_unused=True)
    def norm(values):
        parts = [v.detach().float().square().sum() for v in values if v is not None]
        if not parts:
            return 0.
        value = float(torch.stack(parts).sum().sqrt())
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite reference budget gradient')
        return value
    base_norm, relation_norm = norm(base), norm(relation)
    scale = 1. if relation_norm <= eps else min(1., max(floor, base_norm / (relation_norm + eps)))
    return scale, base_norm, relation_norm


class AdaptivePL(PhysicalPL):
    def __init__(self, c):
        super().__init__(c)
        self._adaptive_ref = None
        self.budget_calls = 0
        self.budget_min = 1.
        self.budget_max = 0.
        self.budget_sum = 0.

    def loss(self, student, device):
        if self._physical_U is not None or self._adaptive_ref is not None:
            raise RuntimeError('Previous adaptive U packet was not consumed')
        try:
            value = HardPLObjective.loss(self, student, device)
            packet, inputs, pseudo, accepted, metadata = self._physical_U
            if set(packet) != {'time', 'frequency', 'pa', 'path', 'reference'}:
                raise ValueError('Five actual PAMF paths required')
            step = student._feature_l_steps + self.c.get('step_offset', 0)
            selected = select_u_candidates(accepted, 128, 202611102 + 2 * step)
            if len(selected):
                directions, norms = physical_directions(student, inputs.index_select(0, selected), 202611104 + 2 * step)
                c0 = dict(self.c, reference_scale=0.)
                nonref, counts = source_relation(packet, directions, pseudo, metadata, selected, c0, accepted=accepted)
                ref, ref_counts = source_relation({'reference': packet['reference']},
                    {'reference': directions['reference']}, pseudo, metadata, selected, self.c, accepted=accepted)
                if counts['valid_anchors'] != ref_counts['valid_anchors']:
                    raise ValueError('Reference and full U relation masks differ')
                detail = {'physical_U_direction_' + k: v.mean() for k, v in norms.items()}
            else:
                nonref = sum(v.sum() * 0. for v in packet.values())
                ref = packet['reference'].sum() * 0.
                counts = dict(valid_anchors=0, mean_distance_correction=nonref.detach())
                detail = {}
            self._adaptive_ref = self.c['physical_relation']['u_weight'] * ref / len(packet)
            self.physical_calls += 1
            self.physical_samples += len(selected)
            self.physical_accepted += int(accepted.index_select(0, selected).sum())
            self.metrics.update(physical_U_loss=(nonref + ref / len(packet)).detach(),
                physical_U_calls=self.physical_calls, physical_U_samples=self.physical_samples,
                physical_U_valid_anchors=counts['valid_anchors'], physical_U_accepted=self.physical_accepted,
                physical_U_distance_correction=counts['mean_distance_correction'], **detail)
            return value + self.c['physical_relation']['u_weight'] * nonref
        except Exception:
            self._adaptive_ref = None
            raise
        finally:
            self._physical_U = None

    def record_budget(self, scale, base_norm, relation_norm):
        self.budget_calls += 1
        self.budget_min = min(self.budget_min, scale)
        self.budget_max = max(self.budget_max, scale)
        self.budget_sum += scale
        self.metrics.update(adaptive_reference_scale=scale, adaptive_reference_base_grad=base_norm,
                            adaptive_reference_relation_grad=relation_norm, adaptive_reference_calls=self.budget_calls)

    def state_dict(self):
        if self._adaptive_ref is not None or self.budget_calls != self.calls:
            raise RuntimeError('Every physical U relation must consume one adaptive budget')
        state = super().state_dict()
        state['adaptive_reference'] = dict(calls=self.budget_calls, minimum=self.budget_min,
            maximum=self.budget_max, mean=self.budget_sum / self.budget_calls if self.budget_calls else None)
        return state


def install_labeled(student, config, holder):
    original_context = student._phase1_labeled_feature_context
    original_hook = student._source_labeled_feature_hook
    student._physical_pending = None
    student._physical_L_calls = student._physical_L_samples = 0

    @contextmanager
    def labeled_context(self):
        if self._physical_pending is not None:
            raise RuntimeError('Unconsumed adaptive L branch packet')
        core, sequence, reference = _actual_paths(self)
        with original_context():
            with capture_branches(core, sequence, reference) as packet:
                yield
        self._physical_pending = packet

    def hook(self, **kwargs):
        packet, self._physical_pending = self._physical_pending, None
        if packet is None or self._cd_pending is None or set(packet) != {'time', 'frequency', 'pa', 'path', 'reference'}:
            raise RuntimeError('Actual adaptive L input or branch packet missing')
        ordinary = self._cd_pending['x']
        value, metrics = original_hook(**kwargs)
        objective = holder['objective']
        u_ref, objective._adaptive_ref = objective._adaptive_ref, None
        if u_ref is None:
            raise RuntimeError('Adaptive U reference relation missing')
        labels = kwargs['labels']
        step = self._feature_l_steps + config.get('step_offset', 0)
        selected = select_indices(torch.ones_like(labels, dtype=torch.bool), 64, 202611101 + 2 * step)
        directions, norms = physical_directions(self, ordinary.index_select(0, selected), 202611103 + 2 * step)
        c0 = dict(config, reference_scale=0.)
        nonref, counts = source_relation(packet, directions, labels, kwargs['extra'][1], selected, c0)
        ref, ref_counts = source_relation({'reference': packet['reference']},
            {'reference': directions['reference']}, labels, kwargs['extra'][1], selected, config)
        if counts['valid_anchors'] != ref_counts['valid_anchors']:
            raise ValueError('Reference and full L relation masks differ')
        reference_loss = config['physical_relation']['l_weight'] * ref / len(packet) + u_ref
        _, _, reference_projection = _actual_paths(self)
        scale, base_norm, relation_norm = gradient_budget(kwargs['identity_loss'] + value, reference_loss,
                                                          reference_projection.parameters())
        objective.record_budget(scale, base_norm, relation_norm)
        self._physical_L_calls += 1
        self._physical_L_samples += len(selected)
        metrics.update(physical_L_loss=(nonref + ref / len(packet)).detach(), physical_L_calls=self._physical_L_calls,
            physical_L_samples=self._physical_L_samples, physical_L_valid_anchors=counts['valid_anchors'],
            physical_L_distance_correction=counts['mean_distance_correction'],
            **{'physical_L_direction_' + k: v.mean() for k, v in norms.items()})
        return value + config['physical_relation']['l_weight'] * nonref + scale * reference_loss, metrics

    hook._accepts_labeled_inputs = True
    student._phase1_labeled_feature_context = MethodType(labeled_context, student)
    student._source_labeled_feature_hook = MethodType(hook, student)
    return student
