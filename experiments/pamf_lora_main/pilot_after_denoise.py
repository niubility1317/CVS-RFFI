"""Learn denoising before fitting a held-out LoRa preamble action."""
import torch

from .channel_blind import ChannelBlind, zero_carrier
from .channel_blind_source import waveform_risk
from .clean_pilot_response import CleanPilotResponse
from .crossfit_pilot import apply_transfer, estimate_shared, generic_taps, regularized_inverse
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import transfer_risk, weak_difference
from .receiver_physics import apply_generic_physics


class PilotAfterDenoise(ChannelBlind):
    def __init__(self, widths, *, denoised_fit):
        super().__init__(widths=widths, companion='mirror')
        self.denoised_fit = denoised_fit
        self.calibration = CleanPilotResponse()

    def pilot_groups(self, observed, quality):
        # Batch the two groups without exposing either to the other's samples.
        inputs = torch.cat((observed[:, ::2], observed[:, 1::2]))
        coherence = torch.cat((quality[:, ::2], quality[:, 1::2]))
        mixed = self.training and observed.is_cuda and torch.cuda.is_bf16_supported()
        groups = self.propose(inputs, coherence, mixed_conv=mixed)
        return tuple(torch.stack(value.chunk(2), dim=2).flatten(1, 2) for value in groups)

    def forward(self, observed, quality, template, *, action=True, diagnostics=False):
        coarse, prior, variance = self.pilot_groups(observed, quality)
        if action:
            fit = coarse if self.denoised_fit else None
            transfer, support = estimate_shared(observed, template, quality, variance, fit_observed=fit)
            response = self.calibration(transfer, support)
        else:
            response = torch.ones_like(observed[:, :1])
            support = observed.real.new_zeros(len(observed))
        fine = regularized_inverse(coarse, prior, template, variance, response)
        return (coarse, fine, support) if diagnostics else (coarse, fine)


def paired_observations(clean, change, perturb, rng, *, with_identity):
    """Two independent noise draws share one generic action and source signal."""
    taps = generic_taps(clean, rng)
    has_echo = torch.rand((len(clean), 1), device=clean.device, generator=rng) < .5
    taps = taps * has_echo
    base, shifted = apply_generic_physics(apply_transfer(clean, taps),
                                           apply_transfer(clean + change, taps), generator=rng)
    first = perturb(base, rng)
    second = perturb(base, rng)
    changed = shifted + (first - base)
    if with_identity:
        first[-2:] = second[-2:] = clean[-2:]
        changed[-2:] = clean[-2:] + change[-2:]
    return torch.stack((first, second, changed), dim=1)


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean = complex_iq(iq)
    framed, rotation, _, rms = coordinates(tracker, iq)
    clean_frame = rotation * rms
    delta = weak_difference(framed)
    change = delta * clean_frame
    wave = paired_observations(clean, change, recipe.perturb, rng, with_identity=with_identity)
    observed, rotation, quality, rms = coordinates(tracker, real_iq(wave.flatten(0, 1)))
    basis = rotation * rms
    coarse, full, support = model(observed, quality, rotation, diagnostics=True)
    coarse, full = recipe.transform_stages(observed, (coarse, full), rms)
    coarse_pair = coarse.unflatten(0, wave.shape[:2])[:, :2]
    paired_target = wave[:, [1, 0]] / basis.unflatten(0, wave.shape[:2])[:, :2]
    same_action = zero_carrier(coarse_pair - paired_target).abs().square().mean()

    targets = torch.stack((clean, clean, clean + change), dim=1)
    fine_risk = waveform_risk(full, targets.flatten(0, 1) / basis)
    half = torch.lerp(wave, (full * basis).unflatten(0, wave.shape[:2]), .5)
    half_risk = waveform_risk(half / clean_frame[:, None], targets / clean_frame[:, None])
    detail = transfer_risk((half[:, 2] - half[:, 0]) / clean_frame, delta)
    weight_coarse, weight_fine, _ = recipe.loss_weights(step)
    loss = weight_coarse*same_action + weight_fine*fine_risk + .30*half_risk + .35*detail
    return loss, torch.stack((same_action, fine_risk, half_risk, detail, support.mean()))
