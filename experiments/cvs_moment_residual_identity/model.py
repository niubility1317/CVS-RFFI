"""CE-learned packet-moment residuals around the existing phase-memory lift.

The two new zero gates retain the own-scratch adaptive lag4 function exactly.
Packet moments are received-IQ coordinates, not identified transmitter PA
coefficients. No packet, class, receiver or target fits any persistent state.
"""
import torch
from torch import nn

from experiments.cvs_adaptive_volterra_identity.model import (
    AdaptiveVolterraCVS, AdaptiveInputBlock, adaptive_basis_from_clipped, adaptive_contract)
from experiments.cvs_coupled_identity.model import coupled_basis_from_clipped
from experiments.cvs_energy_identity.model import GlobalEnergyBlock
from experiments.cvs_orthopoly_identity.model import packet_basis, VARIANCE_FLOOR, MASS_FLOOR

VARIANTS = ('moment_residual_instant', 'moment_residual_memory4')


def moment_basis(z, mix_raw, projection_raw, moment_lag):
    if mix_raw.shape != (2,) or projection_raw.shape != (2,):
        raise ValueError('Expected two phase-memory and two packet-moment gates')
    base = coupled_basis_from_clipped(z, 4)
    adaptive = adaptive_basis_from_clipped(z, 4, mix_raw)
    orthogonal = packet_basis(z, moment_lag)
    scales = z.new_tensor([1., 4., 16.]).repeat(4)
    centered = orthogonal / scales[None, None, :, None]
    coefficients = torch.cat((projection_raw.new_zeros(1), projection_raw.tanh())).repeat(4)
    return adaptive + coefficients[None, None, :, None] * (centered - base)


class MomentInputBlock(AdaptiveInputBlock):
    def __init__(self, block, moment_lag):
        super().__init__(block, 4)
        # Preserve parameters belonging to this own-scratch adaptive model.
        self.mix_raw = block.mix_raw
        self.projection_raw = nn.Parameter(torch.zeros(2))
        self.moment_lag = moment_lag

    def forward(self, basis):
        if basis.ndim != 4 or basis.shape[1:3] != (2, 12):
            raise ValueError('Expected original twelve complex input terms')
        actual = moment_basis(basis[:, :, 0], self.mix_raw, self.projection_raw, self.moment_lag)
        return GlobalEnergyBlock.forward(self, actual)


def moment_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered packet-moment residual candidate')
    lag = 0 if variant == VARIANTS[0] else 4
    contract = adaptive_contract('adaptive_volterra_lag4')
    contract.update(mode=variant, identity_core='moment_residual_identity',
        behavior_input='adaptive lag4 phase-memory lift plus two global tanh-gated packet-centered residuals',
        moment_lag=lag, actual_moment_lag=lag, moment_residual_active=True,
        projection_parameter_shape=[2], actual_projection_parameter_shape=[2],
        projection_initial_raw=[0., 0.], projection_rule='adaptive+tanh(beta)*(unscaled_centered-coupled_lag4)',
        projection_scope='two global model parameters learned only by source CE',
        packet_moments_scope='same received256packet and each delay; no batch pooling, labels, IDs, fitting or persistent state',
        centered_coordinates='z;z*(p-mu);z*((p-mu)^2-a*(p-mu)-variance)',
        centered_envelope='abs(z)^2/4' if lag == 0 else '(abs(z)^2+delay4(abs(z)^2))/8',
        variance_floor=VARIANCE_FLOOR, mass_floor=MASS_FLOOR, per_order_energy_whitening=False,
        projection_parameter_count=2, total_global_residual_gates=4, new_trainable_parameters=2,
        initial_function='exact own-scratch adaptive_volterra_lag4 function under matching RNG; no checkpoint load',
        raw_order1_unchanged=True, complete_projection_is_not_applied=True,
        actual_mixed_inputs_are_not_necessarily_orthogonal=True,
        moments_are_not_streaming_causal=True, hardware_parameter_recovery=False,
        whole_affine_phase_invariant=False)
    return contract


class MomentResidualCVS(AdaptiveVolterraCVS):
    def __init__(self, variant):
        contract = moment_contract(variant)
        super().__init__('adaptive_volterra_lag4')
        self.moment_variant = variant
        self.core.behavior[0] = MomentInputBlock(self.core.behavior[0], contract['moment_lag'])

    def contract(self):
        contract = moment_contract(self.moment_variant)
        block = self.core.behavior[0]
        contract.update(moment_residual_active=isinstance(block, MomentInputBlock),
            actual_moment_lag=getattr(block, 'moment_lag', None),
            actual_phase_lag=getattr(block, 'phase_lag', None),
            actual_envelope_lag=getattr(block, 'envelope_lag', None),
            actual_mixture_parameter_shape=list(block.mix_raw.shape),
            actual_projection_parameter_shape=list(block.projection_raw.shape))
        return contract

    @torch.no_grad()
    def input_diagnostics(self, x):
        records = []
        block = self.core.behavior[0]

        def capture(module, inputs, output):
            actual = inputs[0]
            z = actual[:, :, 0]
            expected = moment_basis(z, block.mix_raw, block.projection_raw, block.moment_lag)
            control = coupled_basis_from_clipped(z, 4)
            adaptive = adaptive_basis_from_clipped(z, 4, block.mix_raw)
            _, moments = packet_basis(z, block.moment_lag, True)
            changes = {}
            for degree, slots in ((3, [1, 4, 7, 10]), (5, [2, 5, 8, 11])):
                denominator = control[:, :, slots].square().sum((1, 2, 3)).sqrt().clamp_min(1e-12)
                delta = (actual[:, :, slots] - control[:, :, slots]).square().sum((1, 2, 3)).sqrt()
                added = (actual[:, :, slots] - adaptive[:, :, slots]).square().sum((1, 2, 3)).sqrt()
                changes['degree'+str(degree)+'_relative_input_change_mean'] = float((delta / denominator).mean())
                changes['degree'+str(degree)+'_moment_residual_relative_change_mean'] = float((added / denominator).mean())
            records.append(dict(block='behavior.0.conv', packets=len(actual), complex_terms=12,
                actual_phase_lag=4, actual_envelope_lag=4, actual_moment_lag=block.moment_lag,
                raw_order1_max_abs_error=float((actual[:, :, [0, 3, 6, 9]] - control[:, :, [0, 3, 6, 9]]).abs().max()),
                input_formula_max_abs_error=float((actual - expected).abs().max()), input_abs_max=float(actual.abs().max()),
                moment_mean_by_delay=[float(m['mean'].mean()) for m in moments],
                moment_variance_by_delay=[float(m['variance'].mean()) for m in moments], **changes))

        hook = block.conv.register_forward_hook(capture)
        try:
            self.features(x)
        finally:
            hook.remove()
        return dict(active=isinstance(block, MomentInputBlock),
            raw_parameters=block.mix_raw.tolist(), coefficients=block.mix_raw.tanh().tolist(),
            projection_raw_parameters=block.projection_raw.tolist(), projection_coefficients=block.projection_raw.tanh().tolist(),
            records=records, scope='Actual received-IQ moment residual input; no TX coefficient identification')

    @torch.no_grad()
    def diagnostics(self, x):
        result = super().diagnostics(x)
        result['moment_input'] = result.pop('adaptive_input')
        return result


def build(variant):
    return MomentResidualCVS(variant)
