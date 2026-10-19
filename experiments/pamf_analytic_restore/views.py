"""Parameter-free original PAMF input transforms; never load F weights or logits."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_identity_ce.model import build_model
from experiments.standard_ce_baselines.sigtime import NoExplicitCFO

NATIVE = build_model.__globals__['CVSincNet']
MemoryPolynomialLift = build_model.__globals__['MemoryPolynomialLift']
SHAPES = dict(frequency=(4, 1), polynomial=(24, 1), statistics=(7,), response=(28,))


class AnalyticViews(nn.Module):
    _to_complex = staticmethod(NATIVE._to_complex)
    _fft_full_via_rfft_pair = NATIVE._fft_full_via_rfft_pair
    _mirror_compressed_features = NATIVE._mirror_compressed_features

    def __init__(self):
        super().__init__()
        self.eps = 1e-8
        self.freq_feature_source = 'raw_fft'
        self.freq_bands = 32
        self.use_rfft_pair = self.use_circularity = self.use_aux_spectral_stats = True
        self.pa_lift = MemoryPolynomialLift(memory_depth=4, orders=(1, 3, 5), clip=2.)
        self.response = NoExplicitCFO('reference_response')

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, 256):
            raise ValueError('Current IQ[B,2,256] required')
        frequency, rho, dac, pa = self._mirror_compressed_features(x)
        response = self.response(x)
        return dict(frequency=frequency, polynomial=self.pa_lift(x),
            statistics=torch.cat((dac, pa, rho), 1), response=torch.cat((response[:, :27], response[:, 28:]), 1))


def scales_from_sums(sums, counts):
    if set(sums) != set(SHAPES) or set(counts) != set(SHAPES):
        raise ValueError('All four analytic groups required')
    result = {}
    for key, shape in SHAPES.items():
        if counts[key] <= 0 or not torch.isfinite(sums[key]).all() or (sums[key] < 0).any():
            raise ValueError('Finite source-only squared sums required')
        result[key] = (sums[key]/counts[key]).sqrt().clamp_min(.01).reshape(shape).float()
    return result


class ViewLoss(nn.Module):
    def __init__(self, scales):
        super().__init__()
        if set(scales) != set(SHAPES):
            raise ValueError('Exactly four fixed source normalization groups required')
        self.views = AnalyticViews()
        for name, shape in SHAPES.items():
            value = scales[name].detach().float() if torch.is_tensor(scales[name]) else torch.tensor(scales[name], dtype=torch.float32)
            if tuple(value.shape) != shape or not torch.isfinite(value).all() or (value < .01).any():
                raise ValueError('Positive source RMS scales with original channel shapes required')
            self.register_buffer(name+'_scale', value.clone())

    def forward(self, estimate, clean):
        predicted = self.views(estimate)
        with torch.no_grad():
            target = self.views(clean)
        parts = {}
        for name, value in predicted.items():
            delta = (value-target[name])/getattr(self, name+'_scale')
            parts[name] = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=1., reduction='none').flatten(1).mean(1)
        return torch.stack(list(parts.values()), 1).mean(1), parts
