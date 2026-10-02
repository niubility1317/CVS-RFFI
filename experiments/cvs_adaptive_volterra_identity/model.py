"""Two CE-trained residual gates for causal received-IQ phase memory.

Zero gates exactly retain the scratch lag4-envelope control. These features
do not identify transmitter hardware coefficients or remove receiver effects.
"""
import torch
from torch import nn
from experiments.cvs_energy_identity.model import EnergyCVS, GlobalEnergyBlock
from experiments.cvs_coupled_identity.model import delay, coupled_basis_from_clipped, coupled_contract
from experiments.cvs_volterra_identity.model import multiply

VARIANTS = ('adaptive_volterra_lag1', 'adaptive_volterra_lag4')


def adaptive_basis_from_clipped(z, lag, mix_raw):
    """B + tanh(a)*(Q-B), separately for third- and fifth-order inputs.

    At t=n-m, B=(z,z*p,z*p^2), Q=(z,C,C*p), p=(P[t]+P[t-4])/8,
    C=z[t-l]^2*conj(z[t-2l])/4. Four causal delays keep twelve inputs.
    Gates are global model parameters, not fitted per packet or receiver.
    """
    if z.ndim != 3 or z.shape[1] != 2 or lag not in (1, 4):
        raise ValueError('Expected complex IQ and fixed phase lag1/4')
    if mix_raw.shape != (2,):
        raise ValueError('Expected exactly two global residual gates')
    base = coupled_basis_from_clipped(z, 4)
    power = z.square().sum(1)
    p = (power + delay(power, 4)) / 8
    history = delay(z, lag)
    conjugate = delay(z, 2 * lag) * z.new_tensor([1., -1.])[None, :, None]
    cubic = multiply(multiply(history, history), conjugate) / 4
    proposal = torch.stack([delay(term, m) for m in range(4)
                            for term in (z, cubic, cubic * p[:, None])], 2)
    coefficients = torch.cat((mix_raw.new_zeros(1), mix_raw.tanh())).repeat(4)
    return base + coefficients[None, None, :, None] * (proposal - base)


class AdaptiveInputBlock(GlobalEnergyBlock):
    def __init__(self, block, lag):
        super().__init__(block)
        self.phase_lag = lag
        self.envelope_lag = 4
        self.mix_raw = nn.Parameter(torch.zeros(2))

    def forward(self, basis):
        if basis.ndim != 4 or basis.shape[1:3] != (2, 12):
            raise ValueError('Expected original twelve-term RF basis')
        return super().forward(adaptive_basis_from_clipped(basis[:, :, 0], self.phase_lag, self.mix_raw))


def adaptive_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown adaptive complex Volterra input candidate')
    lag = 1 if variant == VARIANTS[0] else 4
    d = coupled_contract('coupled_lag4')
    d.pop('coupled_lift_active')
    d.update(mode=variant, behavior_input='lag4 envelope control plus signed tanh-gated delayed cubic residual',
             phase_lag=lag, actual_phase_lag=lag, envelope_lag=4, actual_envelope_lag=4,
             adaptive_lift_active=True, mixture_parameter_shape=[2], actual_mixture_parameter_shape=[2],
             mixture_initial_raw=[0., 0.], mixture_rule='base+tanh(raw)*(proposal-base)',
             mixture_scope='two global identity model parameters trained only by source CE',
             mixture_parameter_count=2, new_trainable_parameters=2,
             initial_function='exact scratch coupled_lag4 function under matching scratch RNG',
             cubic_divisor=4, cubic_delays=[lag, lag, 2 * lag], cubic_conjugate_last=True,
             lift_affine_phase_covariant=True, whole_affine_phase_invariant=False,
             complete_volterra=False, hardware_parameter_recovery=False, prototype_only=False,
             identity_core='adaptive_volterra_identity')
    return d


class AdaptiveVolterraCVS(EnergyCVS):
    def __init__(self, variant):
        contract = adaptive_contract(variant)
        super().__init__('energy_equivariant')
        self.adaptive_variant = variant
        self.core.behavior[0] = AdaptiveInputBlock(self.core.behavior[0], contract['phase_lag'])

    def contract(self):
        d = dict(super().contract(), **adaptive_contract(self.adaptive_variant))
        block = self.core.behavior[0]
        d['adaptive_lift_active'] = isinstance(block, AdaptiveInputBlock)
        d['actual_phase_lag'] = getattr(block, 'phase_lag', None)
        d['actual_envelope_lag'] = getattr(block, 'envelope_lag', None)
        d['actual_mixture_parameter_shape'] = list(block.mix_raw.shape) if hasattr(block, 'mix_raw') else None
        return d

    @torch.no_grad()
    def input_diagnostics(self, x):
        records = []
        block = self.core.behavior[0]

        def capture(module, inputs, output):
            actual = inputs[0]
            z = actual[:, :, 0]
            expected = adaptive_basis_from_clipped(z, block.phase_lag, block.mix_raw)
            control = coupled_basis_from_clipped(z, 4)
            changes = {}
            for degree, slots in ((3, [1, 4, 7, 10]), (5, [2, 5, 8, 11])):
                delta = (actual[:, :, slots] - control[:, :, slots]).square().sum((1, 2, 3)).sqrt()
                base = control[:, :, slots].square().sum((1, 2, 3)).sqrt().clamp_min(1e-12)
                changes['degree' + str(degree) + '_relative_input_change_mean'] = float((delta / base).mean())
            records.append(dict(block='behavior.0.conv', packets=len(actual), complex_terms=actual.shape[2],
                                actual_phase_lag=block.phase_lag, actual_envelope_lag=block.envelope_lag,
                                input_formula_max_abs_error=float((actual - expected).abs().max()),
                                input_abs_max=float(actual.abs().max()), **changes))

        hook = block.conv.register_forward_hook(capture)
        try:
            self.features(x)
        finally:
            hook.remove()
        return dict(scope='Actual adaptive input versus lag4 control; source/public IQ only; not identified TX hardware',
                    active=isinstance(block, AdaptiveInputBlock),
                    raw_parameters=[float(v) for v in block.mix_raw],
                    coefficients=[float(v) for v in block.mix_raw.tanh()], records=records)

    @torch.no_grad()
    def diagnostics(self, x):
        return dict(super().diagnostics(x), adaptive_input=self.input_diagnostics(x))


def build(variant):
    return AdaptiveVolterraCVS(variant)
