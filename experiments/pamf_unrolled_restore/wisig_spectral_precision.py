"""Native WiSig restoration with an estimated frequency-domain observation metric."""

import math

import torch
from torch import nn

from experiments.pamf_action_restore.operators import real_dot
from experiments.pamf_r65_inference.program import reused_map
from experiments.pamf_unrolled_restore.wisig_native_fir import NativeFIR, SCALE, WindowAction, objective
from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex, as_iq, normalized_error


ARMS = ('fixed_precision', 'spectral_precision')


class SpectralWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv1d(2, 8, 5, padding=2, padding_mode='circular')
        self.output = nn.Conv1d(8, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, observed_log_power, error):
        error_log_power = (0.02 + torch.fft.fft(error.detach(), norm='ortho').abs().square()).log()
        features = torch.stack((observed_log_power, error_log_power), dim=1)
        features = features - features.mean(-1, keepdim=True)
        offsets = self.output(torch.nn.functional.silu(self.features(features)))[:, 0].tanh()
        return (math.log(2) * (offsets - offsets.mean(-1, keepdim=True))).exp()


def spectral_apply(weight, value):
    return torch.fft.ifft(weight * torch.fft.fft(value, norm='ortho'), norm='ortho')


def spectral_map(action, prior, initial, variance, error, weight):
    """Two CG iterations on the real-linear, spectrally weighted normal equations."""
    precision = variance.reciprocal()
    delta = initial - prior
    residual = precision * action.adjoint(spectral_apply(weight, error)) - delta / SCALE**2
    direction = residual
    squared = real_dot(residual, residual)
    for step in range(2):
        normal = precision * action.adjoint(spectral_apply(weight, action(direction))) + direction / SCALE**2
        alpha = squared / real_dot(direction, normal).clamp_min(1e-20)
        delta = delta + alpha[:, None] * direction
        if step == 0:
            residual = residual - alpha[:, None] * normal
            next_squared = real_dot(residual, residual)
            direction = residual + (next_squared / squared.clamp_min(1e-20))[:, None] * direction
            squared = next_squared
    return prior + delta


class PrecisionFIR(NativeFIR):
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('Registered WiSig precision arm required')
        super().__init__(centered=True)
        self.spectral_weights = nn.ModuleList(SpectralWeight() for _ in range(2)) if arm == ARMS[1] else None

    def stages(self, analysis):
        value, level, base, kernel, main, image, variance = analysis
        action = WindowAction(kernel, main, image, self.support)
        iq = as_iq(value)
        current = as_complex(base)
        variance = variance.detach()
        logvar = variance.log().clamp(-16, 4)[:, :, None].expand_as(base[:, :1]) / 8
        output = [current * level]
        mixed = self.training and value.is_cuda and torch.cuda.is_bf16_supported()
        observed_log_power = None
        if self.spectral_weights is not None:
            observed_log_power = (0.02 + torch.fft.fft(value.detach(), norm='ortho').abs().square()).log()
        for index, (network, multiplier) in enumerate(zip(self.priors, (8, 1), strict=True)):
            error = value - action(current)
            innovation = action.adjoint(error)
            features = as_iq(innovation)
            features = features / (features.square().mean((1, 2), keepdim=True) + 1e-6).sqrt()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
                update = network(torch.cat((iq, base, as_iq(current), features, logvar), 1)).float()
            prior = as_complex(base + SCALE * update)
            if self.spectral_weights is None:
                current = reused_map(action, prior, current, variance / multiplier, innovation)
            else:
                weight = self.spectral_weights[index](observed_log_power, error)
                current = spectral_map(action, prior, current, variance / multiplier, error, weight)
            output.extend((prior * level, current * level))
        return tuple(output), action


def source_objective(model, source, generator):
    loss, costs = objective(model, source, generator, identity=True)
    clean = source[-4:]
    stages, _ = model.stages(model.analyze(clean))
    clean_error = normalized_error(stages[-1], clean)
    return loss + .20 * clean_error, torch.cat((costs, clean_error[None]))


def check_spectral_map(device):
    torch.manual_seed(17841)
    value, initial, prior = torch.randn(3, 4, 256, device=device, dtype=torch.complex64)
    kernel = torch.randn(4, 81, device=device, dtype=torch.complex64) / 9
    main = torch.ones_like(value) + .1 * value
    image = .1 * value.conj()
    model = PrecisionFIR(ARMS[1]).to(device)
    action = WindowAction(kernel, main, image, model.support)
    error = value - action(initial)
    variance = torch.full((4, 1), .025, device=device)
    weight = torch.ones_like(value.real)
    result = spectral_map(action, prior, initial, variance, error, weight)
    matched = reused_map(action, prior, initial, variance, action.adjoint(error))
    torch.testing.assert_close(result, matched, rtol=5e-5, atol=5e-6)

    metric = torch.exp(torch.linspace(-1, 1, 256, device=device))
    lhs = real_dot(action(initial), spectral_apply(metric, action(prior)))
    rhs = real_dot(action(prior), spectral_apply(metric, action(initial)))
    torch.testing.assert_close(lhs, rhs, rtol=5e-5, atol=5e-5)
