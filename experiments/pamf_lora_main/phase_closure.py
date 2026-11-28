"""Source-paired complex closure loss independent of affine carrier phase."""

import torch

from experiments.pamf_lora_main.w24_native_dura import capture_scale


TILE = 32
LAGS = (1, 3, 7)


def closure_profile(wave):
    normalized = (wave / capture_scale(wave)).unflatten(-1, (-1, TILE))
    # Attenuate phase of weak samples rather than taking their unit phasor.
    soft_phase = normalized / (normalized.abs().square() + .04).sqrt()
    phases = []
    for lag in LAGS:
        middle = soft_phase[..., lag:-lag]
        values = soft_phase[..., :-2*lag] * soft_phase[..., 2*lag:] * middle.conj().square()
        amplitude = normalized[..., lag:-lag].abs()
        phase_moment = values.mean(dim=-1)
        across_amplitude = (amplitude - amplitude.mean(dim=-1,keepdim=True)).tanh()
        amplitude_moment = (across_amplitude * values).mean(dim=-1)
        phases.extend((phase_moment, amplitude_moment))
    return torch.stack(phases, dim=-1)


def phase_closure_loss(wave, clean):
    target = closure_profile(clean).detach()
    error = closure_profile(wave) - target
    mean_error = error.mean(dim=1,keepdim=True)
    private_error = error - mean_error
    centered = target - target.mean(dim=1,keepdim=True)
    common_power = target.abs().square().mean(dim=1,keepdim=True)
    private_power = centered.abs().square().mean(dim=1,keepdim=True)
    common = (mean_error.abs().square()/(.04 + common_power)).log1p().mean()
    private = (private_error.abs().square()/(.04 + private_power)).log1p().mean()
    return .4*common + .6*private
