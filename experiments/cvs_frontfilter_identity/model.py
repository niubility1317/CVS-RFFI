"""Discriminative bounded FIR before every scratch CVS identity path.

Bounds apply to the linear operator with its coefficients held fixed. They do
not establish invertibility of the packet-conditioned nonlinear map, recover a
physical channel, or establish channel/receiver invariance.
"""
import math

import torch
from torch import nn

from experiments.cvs_channel_order_identity.model import (
    BoundedPacketFilter, RHO, _diagnostic_mode, _gradient_norm,
    packet_complex_fir, phase_invariant_context,
)
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS, neural_contract

VARIANTS = ('frontfilter_static', 'frontfilter_dynamic')
BASE_VARIANT = 'neural_residual_shallow'
BASE_PARAMETERS = 220987
NEW_PARAMETERS = dict(zip(VARIANTS, (48, 320)))


class StaticPacketFilter(BoundedPacketFilter):
    """Same random basis, but eight global identity-trained coefficients."""

    def __init__(self):
        nn.Module.__init__(self)
        self.rho = RHO
        self.basis = nn.Parameter(torch.randn(4, 2, 5) / math.sqrt(10))
        self.coeff_raw = nn.Parameter(torch.zeros(2, 4))

    def coefficients(self, x):
        norm = torch.linalg.vector_norm(self.coeff_raw, dim=0).sum()
        return (self.coeff_raw / norm.clamp_min(1.))[None].expand(len(x), -1, -1)


def filter_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered frontfilter architecture')
    dynamic = variant == 'frontfilter_dynamic'
    new = NEW_PARAMETERS[variant]
    d = neural_contract(BASE_VARIANT)
    d.update(
        mode=variant, identity_core='frontfilter_identity', base_variant=BASE_VARIANT,
        base_trainable_parameters=BASE_PARAMETERS, new_trainable_parameters=new,
        total_parameters=BASE_PARAMETERS + new, total_trainable_parameters=BASE_PARAMETERS + new,
        frontfilter_active=True, frontfilter_variant=variant, frontfilter_dynamic=dynamic,
        frontfilter_basis_count=4, frontfilter_taps=5, frontfilter_rho=RHO,
        frontfilter_basis_bound='complex-modulus L1 <= 1',
        frontfilter_coefficient_bound='complex-modulus L1 <= 1',
        frontfilter_context_dimension=8 if dynamic else None,
        frontfilter_context_hidden=16 if dynamic else None,
        frontfilter_context=('current-packet logpower mean/std and lag1/2/4 complex autocorrelation'
                             if dynamic else 'global learned complex coefficients; no input conditioning'),
        frontfilter_exit_initialization='zero',
        frontfilter_boundary='same symmetric two-sample zero padding; no crop; correlation tap convention',
        frontfilter_wiring='NeuralResidualCVS.features(G_x(x)); time, frequency and behavior all receive G_x(x)',
        frontfilter_original_input_bypass=False, frontfilter_auxiliary_branch=False,
        frontfilter_post_unit_rms=False, frontfilter_detach=False,
        frontfilter_fixed_coefficient_singular_value_bounds=[.75, 1.25],
        frontfilter_dynamic_map_invertible_claim=False,
        frontfilter_cross_packet_state=False, frontfilter_inference_fit=False,
        frontfilter_output_samples=256,
        initial_function='exact own scratch neural_residual_shallow; isolated new initialization RNG',
        frontfilter_loss='unchanged single cross_entropy', new_training_strategy=False,
        arbitrary_channel_rx_invariant=False, hardware_parameter_recovery=False,
        interpretation='CE-trained bounded received-waveform filter; not physical channel inversion or TX/RX separation',
    )
    return d


class FrontFilterCVS(NeuralResidualCVS):
    def __init__(self, variant):
        filter_contract(variant)
        super().__init__(BASE_VARIANT)
        self.frontfilter_variant = variant
        with torch.random.fork_rng(devices=[]):
            self.frontfilter = (StaticPacketFilter() if variant == VARIANTS[0] else BoundedPacketFilter())

    def features(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Expected IQ[B,2,256]')
        return super().features(self.frontfilter(x)[0])

    def frontfilter_parameters(self):
        return list(self.frontfilter.parameters())

    def contract(self):
        d = filter_contract(self.frontfilter_variant)
        g = self.frontfilter
        dynamic = self.frontfilter_variant == VARIANTS[1]
        shape_matches = (type(g) is BoundedPacketFilter and len(g.context) == 3
                         and isinstance(g.context[0], nn.Linear)
                         and isinstance(g.context[1], nn.GELU)
                         and isinstance(g.context[2], nn.Linear)
                         and (g.context[0].in_features, g.context[0].out_features,
                              g.context[2].in_features, g.context[2].out_features) == (8, 16, 16, 8)
                         if dynamic else type(g) is StaticPacketFilter and list(g.coeff_raw.shape) == [2, 4])
        ids = {id(p) for p in self.frontfilter_parameters()}
        d.update(
            frontfilter_active=(shape_matches and g.rho == RHO and list(g.basis.shape) == [4, 2, 5]
                                and getattr(self.features, '__func__', None) is FrontFilterCVS.features
                                and self.variant == 'energy_equivariant'
                                and super().contract()['neural_residual_active']),
            neural_residual_active=super().contract()['neural_residual_active'],
            base_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in ids),
            new_trainable_parameters=sum(p.numel() for p in self.frontfilter_parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in self.parameters()),
            total_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        return d

    @torch.no_grad()
    def frontfilter_diagnostics(self, x):
        with _diagnostic_mode(self, x):
            g = self.frontfilter
            gx, coeff = g(x)
            kernel = g.kernel(coeff)
            dynamic = self.frontfilter_variant == VARIANTS[1]
            raw = (g.context(phase_invariant_context(x)).reshape(len(x), 2, 4)
                   if dynamic else g.coeff_raw[None].expand(len(x), -1, -1))
            raw_norm = torch.linalg.vector_norm(raw, dim=1).sum(-1)
            coeff_norm = torch.linalg.vector_norm(coeff, dim=1).sum(-1)
            kernel_norm = torch.linalg.vector_norm(kernel, dim=1).sum(-1)
            raw_basis_norm = torch.linalg.vector_norm(g.basis, dim=1).sum(-1)
            basis_norm = torch.linalg.vector_norm(g.effective_basis(), dim=1).sum(-1)
            input_norm = x.flatten(1).norm(dim=1)
            nonzero = input_norm > 0
            relative = (gx - x).flatten(1).norm(dim=1) / input_norm.clamp_min(1e-12)
            ratios = gx.flatten(1).norm(dim=1)[nonzero] / input_norm[nonzero]
            record = dict(
                block='frontfilter', packets=len(x), relative_input_change_mean=float(relative.mean()),
                relative_input_change_max=float(relative.max()),
                input_norm_ratio_min=float(ratios.min()) if ratios.numel() else None,
                input_norm_ratio_max=float(ratios.max()) if ratios.numel() else None,
                input_norm_ratio_eligible_packets=int(nonzero.sum()),
                coefficient_l1_mean=float(coeff_norm.mean()), coefficient_l1_max=float(coeff_norm.max()),
                kernel_l1_mean=float(kernel_norm.mean()), kernel_l1_max=float(kernel_norm.max()),
                basis_l1_max=float(basis_norm.max()),
                bound_active_fraction=float((raw_norm > 1).float().mean()),
                basis_bound_active_fraction=float((raw_basis_norm > 1).float().mean()),
                coefficient_packet_variance=float(coeff.var(dim=0, unbiased=False).mean()),
                coefficient_l1_by_packet=coeff_norm.tolist(), kernel_l1_by_packet=kernel_norm.tolist(),
                coefficients=coeff.tolist(),
                fixed_coefficient_delta_operator_bound_max=float((g.rho * kernel_norm).max()),
                basis_gradient_norm=_gradient_norm([g.basis]),
                context_gradient_norm=_gradient_norm(g.context.parameters()) if dynamic else None,
                exit_gradient_norm=_gradient_norm(g.context[-1].parameters() if dynamic else [g.coeff_raw]),
            )
        return dict(active=self.contract()['frontfilter_active'], variant=self.frontfilter_variant,
                    dynamic=dynamic, records=[record],
                    scope='Current input x and one actual G_x(x); no post-filter unit RMS, fitting, state or parameter updates; gradients from last CE backward',
                    fixed_coefficient_bound_only=True, dynamic_map_invertible_claim=False)

    @torch.no_grad()
    def diagnostics(self, x):
        with _diagnostic_mode(self, x):
            # Inherited hooks call self.features(x), which filters once. Passing
            # Gx to super here would incorrectly filter twice in those hooks.
            result = super().diagnostics(x)
            result['diagnostic_input_scopes'] = dict(
                coordinate_cfo_and_coherence='original_input_x',
                normalization_adaptive_input_neural_residual='actual_G_x(x), exactly once per diagnostic forward',
                core_weights_and_gradients='parameters and last CE gradients, no input-dependent estimate',
            )
            result['frontfilter'] = self.frontfilter_diagnostics(x)
            return result


def build(variant):
    return FrontFilterCVS(variant)
