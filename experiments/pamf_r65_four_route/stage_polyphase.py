"""Two direct R65 successors, initialized from weights without IQ access.

The polyphase prior changes the prior's time grid and is not function equivalent
to the original prior. Its shuffle preserves every time point within a sample.
"""
import copy

import torch
from torch import nn

from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_iq_joint_compact.models import JointCompactRestorer
from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r46_delayed_prune.models import CompactSuccessor
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_uncertainty_restore.models import weighted_map


STAGE_TARGETS = (1, 2)
PRIOR_WIDTHS = (12, 16, 28, 40)
PARENT_PARAMETERS = 273765


def _parent_or_skeleton(parent):
    supplied = parent is not None
    if parent is None:
        # This path constructs the portable graph for strict checkpoint loading;
        # it does not find or read an external checkpoint.
        parent = CompactSuccessor('prune_after10')
    if (not isinstance(parent, CompactSuccessor)
            or getattr(parent, 'arm', None) != 'prune_after10'
            or len(parent.prior) != 3
            or sum(p.numel() for p in parent.parameters()) != PARENT_PARAMETERS):
        raise ValueError('Only the original compact R65 prune_after10 parent is accepted')
    return parent, supplied


def _report(parent, model, supplied, *, mode, copied, transformed=0):
    total = sum(p.numel() for p in model.parameters())
    if copied + transformed != total:
        raise ValueError('Initialization accounting must cover every unique student parameter')
    return dict(
        mode=mode, parent='original_R65_prune_after10', parent_supplied=supplied,
        construction='parent_weights' if supplied else 'checkpoint_loading_skeleton',
        inherited_initialization=copy.deepcopy(parent.initialization_report),
        parent_parameters=PARENT_PARAMETERS, parameters=total,
        copied_parameters=copied, transformed_parameters=transformed,
        parameter_accounting='unique_destination_scalars_exact_copy_or_transform',
        new_parameters=max(0, total-PARENT_PARAMETERS),
        removed_parameters=max(0, PARENT_PARAMETERS-total), random_parameters=0,
        net_parameter_change=total-PARENT_PARAMETERS,
        new_parameter_initialization='none' if not transformed else 'deterministic_parity_weight_transform',
        IQ_examples=0, TX_labels_read=0, F_reads=0, V_reads=0, target_reads=0,
        optimizer_updates=0, parent_unchanged=True,
        exact_function_equivalence=False, context_trainable=True,
    )


def _stages(priors, received, context):
    """Original stage algorithm, with exactly one iteration per supplied prior."""
    if received.ndim != 3 or received.shape[1:] != (2, 256):
        raise ValueError('Current received IQ2x256 is required')
    base, action, native = context
    variance = native[:, None].expand(-1, 256)
    observed, current, outputs = complex_iq(received), base, []
    logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256)/8
    for prior_core in priors:
        innovation = pair(action.adjoint(observed-action(complex_iq(current))))
        innovation = innovation/(innovation.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
        inputs = torch.cat((received, base, current, innovation, logvar), 1)
        prior = base+SCALE*prior_core(inputs)
        current = pair(weighted_map(action, observed, complex_iq(prior), variance,
                                    steps=2, initial=complex_iq(current)))
        outputs.append(current)
    return outputs, variance, action, base


class StageDistilled(JointCompactRestorer):
    """Two original stages initialized from original stage indices one and two."""

    STAGE_TARGETS = STAGE_TARGETS

    def __init__(self, parent=None):
        nn.Module.__init__(self)
        parent, supplied = _parent_or_skeleton(parent)
        self.arm = 'stage2'
        self.frozen = copy.deepcopy(parent.frozen)
        # One deepcopy preserves aliases between the retained source priors.
        self.prior = copy.deepcopy(nn.ModuleList([parent.prior[i] for i in STAGE_TARGETS]))
        self.requires_grad_(True)
        self.initialization_report = _report(
            parent, self, supplied, mode='R65_last_two_priors_four_CG',
            copied=sum(p.numel() for p in self.parameters()),
        )
        self.initialization_report.update(stage_targets=list(STAGE_TARGETS), stages=2, CG_steps=4)

    def stage_outputs(self, received, context=None):
        if context is None:
            context = self.estimate_context(received)
        return _stages(self.prior, received, context)


def time_unshuffle(value):
    """[B,C,2T] -> [B,2C,T], ordered c0-even,c0-odd,c1-even,... ."""
    if value.ndim != 3 or value.shape[-1] % 2:
        raise ValueError('A three-dimensional tensor with even time length is required')
    return torch.stack((value[..., ::2], value[..., 1::2]), dim=2).flatten(1, 2)


def time_shuffle(value):
    """Inverse of time_unshuffle, without crossing channel or sample boundaries."""
    if value.ndim != 3 or value.shape[1] % 2:
        raise ValueError('A three-dimensional tensor with even channel count is required')
    batch, channels, length = value.shape
    return value.reshape(batch, channels//2, 2, length).transpose(2, 3).reshape(batch, channels//2, 2*length)


class PolyphasePrior(nn.Module):
    """A half-rate Core over reversible even/odd channels, with full-rate IQ out."""

    def __init__(self, source=None):
        super().__init__()
        self.core = Core(incoming=18, outgoing=4, widths=PRIOR_WIDTHS)
        self.exact_copied_parameters = 0
        self.transformed_parameters = 0
        if source is not None:
            self._initialize_from(source)

    def _initialize_from(self, source):
        if not isinstance(source, Core):
            raise ValueError('Original R65 prior Core is required')
        self.core.to(next(source.parameters()))
        original = dict(source.named_parameters())
        source_state = source.state_dict()
        input_weights = {'encoder.blocks.0.0.net.0.weight', 'encoder.blocks.0.0.skip.weight'}
        output_weights = {'out.weight', 'out.bias'}
        with torch.no_grad():
            for name, value in self.core.named_parameters():
                if name not in original:
                    raise ValueError('R65 prior parameter layout differs')
                old = source_state[name]
                if name in input_weights:
                    initial = old.repeat_interleave(2, dim=1)/2
                elif name in output_weights:
                    initial = old.repeat_interleave(2, dim=0)
                elif value.shape == old.shape:
                    initial = old
                    self.exact_copied_parameters += value.numel()
                else:
                    raise ValueError('Unexpected R65 prior parameter shape')
                if initial.shape != value.shape:
                    raise ValueError('Parity transform does not match the R65 prior shape')
                if name in input_weights | output_weights:
                    self.transformed_parameters += value.numel()
                value.copy_(initial)

    def forward(self, value):
        if value.ndim != 3 or value.shape[1:] != (9, 256):
            raise ValueError('Current nine-channel full IQ-window prior input is required')
        return time_shuffle(self.core(time_unshuffle(value)))


class PolyphaseRestorer(JointCompactRestorer):
    """Original full-rate context and three two-CG stages with half-rate priors."""

    STAGE_TARGETS = (0, 1, 2)

    def __init__(self, parent=None):
        nn.Module.__init__(self)
        parent, supplied = _parent_or_skeleton(parent)
        self.arm = 'polyphase'
        self.frozen = copy.deepcopy(parent.frozen)
        self.prior = nn.ModuleList(PolyphasePrior(source) for source in parent.prior)
        # Restore all original long-convolution aliases after parity conversion.
        canonical = {name: module.weight.detach().clone()
                     for name, module in self.prior[0].named_modules()
                     if isinstance(module, nn.Conv1d) and module.kernel_size[0] > 1}
        share_long_weights(self.prior)
        # The helper averages identical source aliases. Recopy the canonical
        # values to avoid roundoff from that average changing inherited weights.
        with torch.no_grad():
            modules = dict(self.prior[0].named_modules())
            for name, weight in canonical.items():
                modules[name].weight.copy_(weight)
        self.requires_grad_(True)
        transformed_names = {
            'encoder.blocks.0.0.net.0.weight', 'encoder.blocks.0.0.skip.weight',
            'out.weight', 'out.bias',
        }
        # Count shared tensors once, as model.parameters() does.
        transformed = sum(p.numel() for name, p in self.named_parameters()
                          if name.startswith('prior.') and name.split('.core.', 1)[-1] in transformed_names)
        total = sum(p.numel() for p in self.parameters())
        self.initialization_report = _report(
            parent, self, supplied, mode='R65_reversible_even_odd_half_rate_priors',
            copied=total-transformed, transformed=transformed,
        )
        self.initialization_report.update(
            stage_targets=list(self.STAGE_TARGETS), stages=3, CG_steps=6,
            prior_time_points=128, input_time_points=256,
            phase_order='per_channel_even_then_odd', whole_IQ_window=True,
        )

    def stage_outputs(self, received, context=None):
        if context is None:
            context = self.estimate_context(received)
        return _stages(self.prior, received, context)
