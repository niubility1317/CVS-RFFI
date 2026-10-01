"""Causal RF behavior operators, with separate physical and recognition claims."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from experiments.cvs_residual_identity.model import build as residual_build

VARIANTS = ('rf_mp', 'rf_gmp')
ALIGNED = tuple((p, m, 0) for m in range(4) for p in (1, 3, 5))
CROSS = tuple((p, m, r) for m in range(4) for r in (1, 2) for p in (3, 5))


def operator_contract(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered RF operator variant')
    return dict(physics_mode='causal_complex_mp' if variant == 'rf_mp' else 'causal_complex_lag_gmp',
                orders=[1,3,5], aligned_delays=[0,1,2,3],
                cross_envelope_lags=[1,2] if variant == 'rf_gmp' else [],
                basis_terms=12 if variant == 'rf_mp' else 28,
                complex_channels=8, observation_channels=32, radial_clip=4.0,
                epsilon=1e-6, order_scale_base=2.0, sample_rate_hz=25000000,
                input='received_equalized1_unit_rms_IQ', coefficient_interpretation='shared_source_learned_behavior_filters')


def causal_delay(x, delay):
    if delay == 0:
        return x
    if delay >= x.shape[-1]:
        return torch.zeros_like(x)
    return F.pad(x[..., :-delay], (delay, 0))


class CausalRFOperator(nn.Module):
    """Truncated received-IQ behavior bank, not identified TX coefficients.

    D uses phi, V uses conjugate(phi). Under a constant input phase their
    charges are +1/-1. Classifier observations have charge zero. No claim
    about CFO, arbitrary multipath, or the unconstrained other CVS paths.
    """
    def __init__(self, variant, channels=8):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unregistered RF operator variant')
        self.variant = variant
        self.channels = channels
        self.eps = 1e-6
        self.radial_clip = 4.0
        self.basis_spec = ALIGNED + (CROSS if variant == 'rf_gmp' else ())
        self.physics_mode = 'causal_complex_mp' if variant == 'rf_mp' else 'causal_complex_lag_gmp'
        # Both candidates draw the same full scratch bank, then retain only
        # their registered terms. Common terms and downstream RNG match.
        full = torch.randn(2, 2, channels, len(ALIGNED) + len(CROSS)) / math.sqrt(2 * (len(ALIGNED) + len(CROSS)))
        self.coefficients = nn.Parameter(full[..., :len(self.basis_spec)].clone())

    def basis(self, iq):
        if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[-1] < 1:
            raise ValueError('Expected received IQ [B,2,T] with T>=1')
        power = iq.square().sum(1, keepdim=True)
        scale = (self.radial_clip * torch.rsqrt(power + self.eps)).clamp(max=1.0)
        z = iq * scale
        power = z.square().sum(1)
        real, imag = z[:, 0], z[:, 1]
        terms = []
        for p, m, r in self.basis_spec:
            envelope = causal_delay(power, m + r).pow((p - 1) // 2) / (2.0 ** (p - 1))
            terms.append(torch.stack((causal_delay(real, m) * envelope,
                                      causal_delay(imag, m) * envelope), dim=1))
        return torch.stack(terms, dim=2)  # [B,2,basis,T]

    def components(self, iq):
        phi = self.basis(iq)
        # One real Conv1d for complex multiplication with tied signs. Its MAC
        # count is measured by the common profiler. No bias or future taps.
        a, b = self.coefficients
        ar, ai = a[0], a[1]
        br, bi = b[0], b[1]
        weight = torch.cat((torch.cat((ar, -ai), dim=1),
                            torch.cat((ai, ar), dim=1),
                            torch.cat((br, bi), dim=1),
                            torch.cat((bi, -br), dim=1)), dim=0).unsqueeze(-1)
        out = F.conv1d(torch.cat((phi[:, 0], phi[:, 1]), dim=1), weight)
        return out.split(self.channels, dim=1)  # Dr,Di,Vr,Vi

    def forward(self, iq):
        dr, di, vr, vi = self.components(iq)
        pd = dr.square() + di.square()
        pv = vr.square() + vi.square()
        denom = torch.sqrt((pd + self.eps) * (pv + self.eps))
        return torch.cat((torch.log1p(pd), torch.log1p(pv),
                          (dr * vr - di * vi) / denom,
                          (dr * vi + di * vr) / denom), dim=1)

    @torch.no_grad()
    def diagnostics(self):
        w = self.coefficients
        g = w.grad
        result = dict(physics_mode=self.physics_mode, basis_terms=len(self.basis_spec),
                      direct_coefficient_norm=float(w[0].norm()), image_coefficient_norm=float(w[1].norm()),
                      direct_gradient_norm=float(g[0].norm()) if g is not None else None,
                      image_gradient_norm=float(g[1].norm()) if g is not None else None)
        result['order_coefficient_norm'] = {str(p): float(w[..., [j for j, spec in enumerate(self.basis_spec) if spec[0] == p]].norm()) for p in (1, 3, 5)}
        cross = [j for j, spec in enumerate(self.basis_spec) if spec[2] > 0]
        result['cross_memory_coefficient_norm'] = float(w[..., cross].norm()) if cross else None
        return result

    def contract(self):
        result = operator_contract(self.variant)
        result.update(complex_channels=self.channels, observation_channels=4*self.channels,
                      radial_clip=self.radial_clip, epsilon=self.eps,
                      basis_terms=len(self.basis_spec), physics_mode=self.physics_mode)
        return result


class PassEnvelope(nn.Module):
    def forward(self, observations, raw_iq):
        return observations


def build(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered RF operator variant')
    model = residual_build('residual_fusion')
    b = model.id_backbone
    if b.pa_feature_source != 'raw_iq' or b.time_stability is not None or b.freq_stability is not None:
        raise ValueError('RF operator requires the registered raw PA and original residual framework')
    b.pa_lift = CausalRFOperator(variant)
    b.pa_gate = PassEnvelope()
    conv = b.pa_b1.conv
    b.pa_b1.conv = nn.Conv1d(32, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             dilation=conv.dilation, bias=False)
    return model
