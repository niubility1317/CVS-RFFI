"""Bounded packet compensation and nonlinear operation-order interaction.

Every variant keeps the full scratch shallow CVS backbone and original head.
The auxiliary path is an additive structure control, not full-backbone input
equalization, a recovered channel inverse, or an arbitrary RX invariant.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as functional

from experiments.cvs_equivariant_identity.model import ComplexConv, InvariantReadout
from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS, neural_contract


VARIANTS = ('channel_capacity', 'channel_compensated', 'channel_dual', 'channel_order')
BASE_VARIANT = 'neural_residual_shallow'
BASE_PARAMETERS = 220987
NEW_PARAMETERS = dict(zip(VARIANTS, (26464, 13600, 26400, 28960)))
CROP = 5
RHO = .25


def phase_invariant_context(x):
    """Eight current-packet power/autocorrelation summaries, no fitted state."""
    real, imag = x.unbind(1)
    power = real.square() + imag.square()
    logpower = torch.log1p(power)
    values = [logpower.mean(-1), (logpower.var(-1, unbiased=False) + 1e-6).sqrt()]
    for lag in (1, 2, 4):
        ar, ai = real[:, lag:], imag[:, lag:]
        br, bi = real[:, :-lag], imag[:, :-lag]
        scale = ((ar.square() + ai.square()).mean(-1)
                 * (br.square() + bi.square()).mean(-1) + 1e-6).sqrt()
        values.extend(((ar * br + ai * bi).mean(-1) / scale,
                       (ai * br - ar * bi).mean(-1) / scale))
    return torch.stack(values, 1)


def packet_complex_fir(z, kernel):
    """Same zero padding, one Bx2x5 complex FIR shared across packet channels."""
    if z.ndim != 4 or z.shape[1] != 2 or kernel.shape != (len(z), 2, 5):
        raise ValueError('Expected z[B,2,C,T] and packet FIR[B,2,5]')
    windows = functional.pad(z, (2, 2)).unfold(-1, 5, 1)
    real, imag = windows.unbind(1)
    kr, ki = kernel.unbind(1)
    kr, ki = kr[:, None, None, :], ki[:, None, None, :]
    return torch.stack(((real * kr - imag * ki).sum(-1),
                        (real * ki + imag * kr).sum(-1)), 1)


class BoundedPacketFilter(nn.Module):
    """G_x=I+rho*sum_k(a_k B_k), with two complex-modulus L1 bounds."""

    def __init__(self):
        super().__init__()
        self.rho = RHO
        self.basis = nn.Parameter(torch.randn(4, 2, 5) / math.sqrt(10))
        self.context = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        with torch.no_grad():
            self.context[-1].weight.zero_()
            self.context[-1].bias.zero_()

    def effective_basis(self):
        norm = torch.linalg.vector_norm(self.basis, dim=1).sum(-1)
        return self.basis / norm.clamp_min(1.)[:, None, None]

    def coefficients(self, x):
        raw = self.context(phase_invariant_context(x)).reshape(len(x), 2, 4)
        norm = torch.linalg.vector_norm(raw, dim=1).sum(-1)
        return raw / norm.clamp_min(1.)[:, None, None]

    def kernel(self, coefficients):
        basis = self.effective_basis()
        ar, ai = coefficients.unbind(1)
        br, bi = basis.unbind(1)
        return torch.stack((ar @ br - ai @ bi, ar @ bi + ai @ br), 1)

    def apply_filter(self, z, coefficients):
        return z + self.rho * packet_complex_fir(z, self.kernel(coefficients))

    def forward(self, x):
        coefficients = self.coefficients(x)
        return self.apply_filter(x[:, :, None, :], coefficients)[:, :, 0], coefficients


class NonlinearRadial(nn.Module):
    """A genuinely nonlinear, unnormalized, phase-equivariant radial map."""

    def __init__(self, channels):
        super().__init__()
        self.slope = nn.Parameter(torch.full((channels,), .25))

    def forward(self, z):
        power = z.square().sum(1)
        return z * (1 + self.slope[None, :, None] * power.tanh())[:, None]


class LocalComplexFeature(nn.Sequential):
    def __init__(self):
        super().__init__(ComplexConv(1, 8, 5), NonlinearRadial(8),
                         ComplexConv(8, 8, 3), NonlinearRadial(8))


def cross_order_readout(u, difference):
    """Linear in D at D=0; denominator uses u only, never D's magnitude."""
    ur, ui = u.unbind(1)
    dr, di = difference.unbind(1)
    denominator = (ur.square() + ui.square()).mean(-1).clamp_min(1e-6)
    return torch.cat(((ur * dr + ui * di).mean(-1) / denominator,
                      (ur * di - ui * dr).mean(-1) / denominator), 1)


class ChannelBranch(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.feature = LocalComplexFeature()
        self.readout = InvariantReadout()
        self.compensation = None
        self.extra = None
        self.last_d_gradient_norm = None
        if variant == 'channel_capacity':
            self.extra = ComplexConv(8, 8, 3)
        else:
            # Matching F initialization is independent of the compensation arm.
            with torch.random.fork_rng(devices=[]):
                self.compensation = BoundedPacketFilter()
        self.u_project = None if variant == 'channel_compensated' else nn.Linear(80, 160, bias=False)
        self.v_project = nn.Linear(80, 160, bias=False)
        self.d_project = nn.Linear(16, 160, bias=False) if variant == 'channel_order' else None
        with torch.no_grad():
            if self.u_project is not None:
                self.u_project.weight.zero_()
            self.v_project.weight.zero_()
            if self.d_project is not None:
                self.d_project.weight.normal_(std=.001)

    def _d_gradient(self, gradient):
        # Scalar training evidence only; no activation cache or inference update.
        self.last_d_gradient_norm = float(gradient.detach().norm())

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Expected IQ[B,2,256]')
        u = v = w = difference = gx = coefficients = None
        if self.compensation is None:
            u = self.feature(x[:, :, None, :])
            v = self.extra(u)
        else:
            gx, coefficients = self.compensation(x)
            v = self.feature(gx[:, :, None, :])
            if self.u_project is not None:
                u = self.feature(x[:, :, None, :])
            if self.d_project is not None:
                # Reuse the same coefficients and same temporal sampling rate.
                w = self.compensation.apply_filter(u, coefficients)
        u = None if u is None else u[..., CROP:-CROP]
        v = v[..., CROP:-CROP]
        w = None if w is None else w[..., CROP:-CROP]
        residual = self.v_project(self.readout(v))
        if self.u_project is not None:
            residual = residual + self.u_project(self.readout(u))
        if self.d_project is not None:
            difference = v - w
            if difference.requires_grad:
                difference.register_hook(self._d_gradient)
            residual = residual + self.d_project(cross_order_readout(u, difference))
        return dict(residual=residual, u=u, v=v, w=w, difference=difference,
                    compensated_input=gx, coefficients=coefficients)


def channel_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered channel-order architecture')
    new_count = NEW_PARAMETERS[variant]
    compensation = variant != 'channel_capacity'
    order = variant == 'channel_order'
    d = neural_contract(BASE_VARIANT)
    d.update(
        mode=variant, identity_core='channel_order_identity', base_variant=BASE_VARIANT,
        base_trainable_parameters=BASE_PARAMETERS, new_trainable_parameters=new_count,
        total_parameters=BASE_PARAMETERS + new_count, total_trainable_parameters=BASE_PARAMETERS + new_count,
        channel_active=True, channel_compensation_active=compensation, channel_order_active=order,
        compensation_active=compensation, order_difference_active=order,
        channel_rho=RHO if compensation else None, channel_basis_count=4 if compensation else None,
        channel_filter_taps=5 if compensation else None,
        channel_basis_bound='complex-modulus L1 <= 1' if compensation else None,
        channel_coefficient_bound='complex-modulus L1 <= 1' if compensation else None,
        channel_context_dimension=8 if compensation else None,
        channel_context_hidden=16 if compensation else None,
        channel_context_exit_initialization='zero' if compensation else None,
        channel_context='current-packet logpower mean/std and lag1/2/4 complex autocorrelation' if compensation else None,
        channel_feature_channels=[1, 8, 8], channel_feature_kernels=[5, 3],
        channel_feature_nonlinearity='z*(1+slope*tanh(complex_power)); initial slope=.25',
        channel_feature_normalization=False, channel_feature_bias=False, channel_feature_downsampling=False,
        channel_shared_feature=True, channel_boundary='symmetric zero padding then common valid crop',
        channel_valid_crop=CROP, channel_valid_tokens=246,
        channel_difference='F(G_x x)-G_x F(x); same coefficients and sample rate' if order else None,
        channel_difference_readout='mean(conj(u)*D)/mean(abs(u)^2), real and imaginary, linear in D' if order else None,
        channel_difference_exit_initialization='normal std=.001; nonzero' if order else None,
        channel_retains_base_features=True, channel_full_backbone_preequalization=False,
        channel_output_dimension=160,
        channel_fusion='base.features(x)+auxiliary_residual; original classifier applies one final L2 normalization',
        channel_capacity_control='F(x), ordinary complex8to8 k3, two invariant readouts' if not compensation else None,
        channel_inference_fit=False, channel_cross_packet_state=False,
        initial_function='exact own scratch neural_residual_shallow logits; isolated new initialization RNG',
        channel_loss='unchanged single cross_entropy', new_training_strategy=False,
        channel_interpretation='bounded discriminative auxiliary compensation; not true inverse, TX-only feature, or channel invariance',
    )
    return d


def _gradient_norm(parameters):
    gradients = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else None


@contextmanager
def _diagnostic_mode(model, x):
    flags = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=[x.get_device()] if x.is_cuda else []):
            model.eval()
            yield
    finally:
        for module, training in flags:
            module.training = training


class ChannelOrderCVS(NeuralResidualCVS):
    def __init__(self, variant):
        channel_contract(variant)
        super().__init__(BASE_VARIANT)
        self.channel_variant = variant
        with torch.random.fork_rng(devices=[]):
            self.channel_branch = ChannelBranch(variant)

    def features(self, x):
        return super().features(x) + self.channel_branch(x)['residual']

    def channel_parameters(self):
        return list(self.channel_branch.parameters())

    def contract(self):
        d = channel_contract(self.channel_variant)
        branch = self.channel_branch
        compensation = isinstance(branch.compensation, BoundedPacketFilter)
        order = isinstance(branch.d_project, nn.Linear)
        expected_compensation = self.channel_variant != 'channel_capacity'
        new_ids = {id(p) for p in self.channel_parameters()}
        feature_matches = (isinstance(branch.feature, LocalComplexFeature)
                           and [(block.cin, block.cout, block.k, block.stride, block.dilation)
                                for block in branch.feature if isinstance(block, ComplexConv)]
                           == [(1, 8, 5, 1, 1), (8, 8, 3, 1, 1)]
                           and isinstance(branch.feature[1], NonlinearRadial)
                           and isinstance(branch.feature[3], NonlinearRadial))
        g_matches = (not expected_compensation or
                     (compensation and branch.compensation.rho == RHO
                      and list(branch.compensation.basis.shape) == [4, 2, 5]
                      and [(m.in_features, m.out_features) for m in branch.compensation.context
                           if isinstance(m, nn.Linear)] == [(8, 16), (16, 8)]))
        d.update(
            neural_residual_active=super().contract()['neural_residual_active'],
            channel_active=(isinstance(branch, ChannelBranch) and feature_matches and g_matches
                            and compensation == expected_compensation and order == (self.channel_variant == 'channel_order')
                            and branch.variant == self.channel_variant),
            channel_compensation_active=compensation, channel_order_active=order,
            compensation_active=compensation, order_difference_active=order,
            base_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad and id(p) not in new_ids),
            new_trainable_parameters=sum(p.numel() for p in self.channel_parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in self.parameters()),
            total_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        return d

    @torch.no_grad()
    def channel_diagnostics(self, x):
        captured = []
        hook = self.channel_branch.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            with _diagnostic_mode(self, x):
                features = self.features(x)
        finally:
            hook.remove()
        actual = captured[0]
        branch = self.channel_branch
        compensation = branch.compensation is not None
        order = branch.d_project is not None
        residual = actual['residual']
        base = features - residual
        result = dict(
            active=self.contract()['channel_active'], variant=self.channel_variant, packets=len(x),
            compensation_active=compensation, order_difference_active=order,
            g_magnitude_mean=None, g_magnitude_min=None, g_magnitude_max=None,
            g_operator_delta_bound_max=None, g_coefficient_l1_max=None, g_basis_l1_max=None,
            g_context_gradient_norm=None, g_context_exit_gradient_norm=None, g_basis_gradient_norm=None,
            d_relative_output_mean=None, d_relative_output_max=None, d_gradient_norm=None,
            d_readout_gradient_norm=None, d_readout_norm=None,
            f_gradient_norm=_gradient_norm(branch.feature.parameters()),
            residual_relative_output_mean=float((residual.norm(dim=1) / base.norm(dim=1).clamp_min(1e-12)).mean()),
            valid_crop=CROP, valid_tokens=actual['v'].shape[-1],
            scope='current source batch forward; parameter and D activation gradients from last CE backward; no fitting or updates',
        )
        if compensation:
            g = branch.compensation
            change = (actual['compensated_input'] - x).flatten(1).norm(dim=1) / x.flatten(1).norm(dim=1).clamp_min(1e-12)
            coefficient_norm = torch.linalg.vector_norm(actual['coefficients'], dim=1).sum(-1)
            basis_norm = torch.linalg.vector_norm(g.effective_basis(), dim=1).sum(-1)
            bound = RHO * (torch.linalg.vector_norm(actual['coefficients'], dim=1) * basis_norm[None]).sum(-1)
            result.update(g_magnitude_mean=float(change.mean()), g_magnitude_min=float(change.min()),
                          g_magnitude_max=float(change.max()), g_operator_delta_bound_max=float(bound.max()),
                          g_coefficient_l1_max=float(coefficient_norm.max()), g_basis_l1_max=float(basis_norm.max()),
                          g_context_gradient_norm=_gradient_norm(g.context.parameters()),
                          g_context_exit_gradient_norm=_gradient_norm(g.context[-1].parameters()),
                          g_basis_gradient_norm=_gradient_norm([g.basis]))
        if order:
            ratio = actual['difference'].flatten(1).norm(dim=1) / actual['u'].flatten(1).norm(dim=1).clamp_min(1e-12)
            result.update(d_relative_output_mean=float(ratio.mean()), d_relative_output_max=float(ratio.max()),
                          d_gradient_norm=branch.last_d_gradient_norm,
                          d_readout_gradient_norm=_gradient_norm(branch.d_project.parameters()),
                          d_readout_norm=float(branch.d_project.weight.norm()))
        return result

    @torch.no_grad()
    def diagnostics(self, x):
        with _diagnostic_mode(self, x):
            return dict(super().diagnostics(x), channel_order=self.channel_diagnostics(x))


def build(variant):
    return ChannelOrderCVS(variant)
