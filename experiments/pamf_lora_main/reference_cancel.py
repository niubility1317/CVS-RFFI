"""Cross-fitted complex interference cancellation on the received LoRa preamble."""

import torch

from .channel_blind import corruption_score
from .channel_blind_source import clean_threshold, waveform_risk
from .data import SOURCE
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .lapped_posterior import LappedPosterior
from .local_phase import generic_phase_observation
from .natural_detail import paired_inputs, transfer_risk, weak_difference


def bandwidths():
    return (torch.arange(1, 9), torch.arange(9, 33))


def pair_spectrum(wave):
    spectrum = torch.fft.fft(wave, norm='ortho').unflatten(1, (3, 2))
    left, right = spectrum.unbind(2)
    return torch.stack((.5 * (left + right), .5 * (left - right)), dim=2)


def band_observations(spectrum, bins):
    target = spectrum[..., bins % 256]
    # At least 33 kHz from the dechirped carrier and no bins used in the target.
    distance = bins.abs()
    index = bins.sign()[:, None] * torch.stack((distance + 32, distance + 64, distance + 95), dim=-1)
    return target, spectrum[..., index % 256]


def project_reference(value, direction):
    projection = (value * direction).sum(-1, keepdim=True)
    return value - projection * direction.conj()


def rotate_folds(value):
    # For deployment on pair k, estimate on k+1 and verify on k+2.
    return value.roll(-1, dims=1), value.roll(-2, dims=1)


def ridge_reference(observed, reference):
    count = observed.shape[-1]
    gram = (reference.mH @ reference) / count
    cross = (reference.mH @ observed.unsqueeze(-1)) / count
    scale = gram.diagonal(dim1=-1, dim2=-2).real.mean(-1)
    ridge = .01 * scale + 1e-7
    regularizer = torch.eye(reference.shape[-1], dtype=reference.dtype, device=reference.device)
    coefficients = torch.linalg.solve_ex(gram + ridge[..., None, None] * regularizer, cross).result
    return coefficients.squeeze(-1), scale > 1e-7


def read_relative_action(value, quality):
    weight = .1 + .9 * quality
    others = (value * weight).sum(1, keepdim=True) - value * weight
    reference = others / (weight.sum(1, keepdim=True) - weight).clamp_min(1e-8)
    # One local fit per 32-sample step. Fit only the relative, low-order gain.
    v = value.reshape(*value.shape[:2], 8, 32)
    r = reference.reshape_as(v)
    power = r.abs().square().mean(-1)
    disagreement = (v - r).abs().square().mean(-1)
    cross = (v * r.conj()).mean(-1)
    confidence = quality * power / (.05 + power + disagreement)
    action = 1. + confidence * (cross - power) / (.15 + power)
    return action, confidence, disagreement


class ReferenceCancel(LappedPosterior):
    def __init__(self, guarded):
        super().__init__(local=False)
        self.guarded = guarded
        for name, bins in zip(('near', 'middle'), bandwidths(), strict=True):
            signed = torch.cat((bins, -bins))
            self.register_buffer(f'{name}_bins', signed, persistent=False)
            self.register_buffer(f'{name}_center', torch.zeros(2, len(signed), dtype=torch.complex64))
            self.register_buffer(f'{name}_reference', torch.zeros(2, len(signed), 3, dtype=torch.complex64))
            self.register_buffer(f'{name}_blocked', torch.zeros(2, 1, 3, dtype=torch.complex64))
            self.register_buffer(f'{name}_budget', torch.zeros(2))
            self.register_buffer(f'{name}_margin', torch.zeros(2))
            self.register_buffer(f'{name}_clean_leakage', torch.zeros(2))
        self.register_buffer('training_gate', torch.tensor(0.0))
        self.register_buffer('calibrated_noise', torch.tensor(0.0))

    def signed_residual(self, spectrum, name):
        target, reference = band_observations(spectrum, getattr(self, f'{name}_bins'))
        target = target - getattr(self, f'{name}_center')
        reference = reference - getattr(self, f'{name}_reference')
        blocked = getattr(self, f'{name}_blocked')
        return target, project_reference(reference, blocked)

    def band_prediction(self, spectrum, name):
        measured, reference = self.signed_residual(spectrum, name)
        fit, check = rotate_folds(reference)
        fit_target, check_target = rotate_folds(measured)
        coefficients, supported = ridge_reference(fit_target, fit)
        prediction = (reference * coefficients.unsqueeze(-2)).sum(-1)
        predicted_check = (check * coefficients.unsqueeze(-2)).sum(-1)
        value = predicted_check.abs().square().mean(-1)
        correlation = (check_target.conj() * predicted_check).mean(-1).real
        allowance = getattr(self, f'{name}_budget')
        excess = correlation - (allowance * value).sqrt() - .025 * (allowance + value)
        return prediction, supported, excess, value

    @torch.no_grad()
    def cancellation(self, y, enabled=True):
        active = (corruption_score(y) > self.training_gate)[:, None, None]
        spectrum = pair_spectrum(y)
        update = torch.zeros_like(spectrum)
        gains = []
        for name in ('near', 'middle'):
            prediction, supported, excess, value = self.band_prediction(spectrum, name)
            if self.guarded:
                margin = getattr(self, f'{name}_margin')
                dose = ((excess - margin) / (value + 1e-8)).clamp(0., 1.)
            else:
                dose = torch.ones_like(value)
            dose = dose * supported * enabled * active
            gains.append(dose)
            update[..., getattr(self, f'{name}_bins') % 256] = -prediction * dose.unsqueeze(-1)
        common, private = update.unbind(2)
        spectrum_edit = torch.stack((common + private, common - private), dim=2).flatten(1, 2)
        return torch.fft.ifft(spectrum_edit, norm='ortho'), torch.stack(gains, dim=-1)

    def read(self, y, quality, level, *, use_action=True, use_noise=True, use_cancel=True):
        action, confidence, disagreement = read_relative_action(y, quality)
        previous = torch.cat((action[..., :1], action[..., :-1]), dim=-1)
        lapped_action = .5 * (action + previous)
        edit, dose = self.cancellation(y, use_cancel)
        noise = None if use_noise else self.calibrated_noise.reshape(1, 1, 1, 1)
        coarse, fine, prior, variance = self.restore(
            y, quality, reference_action=lapped_action if use_action else None, noise_override=noise,
            observation_edit=edit,
        )
        results = []
        for estimate in (coarse, fine):
            correction = estimate - y
            # Enforce the dose in original IQ units after fixing the carrier.
            power = (correction * level).abs().square().mean((1, 2), keepdim=True)
            observed_power = (y * level).abs().square().mean((1, 2), keepdim=True)
            cap = (.01 * observed_power / power.clamp_min(1e-12)).clamp_max(1.).sqrt()
            results.append(y + cap * correction)
        statistics = (dose, confidence, disagreement, edit.abs().square().mean(dim=(1, 2)).sqrt())
        return *results, prior, variance, statistics


def generic_observation(wave, rng):
    """Add a generic off-bin interferer with common and independent phase."""
    noise = generic_phase_observation(wave, rng)
    batch, chirps, size = wave.shape
    def draw(*shape):
        return torch.rand(shape, device=wave.device, generator=rng)
    time = torch.arange(size, device=wave.device)
    frequency = draw(batch, 1, 1) - .5
    common = draw(batch, 1, 1)
    independent = draw(batch, chirps, 1)
    mix = draw(batch, 1, 1)
    phase = 2 * torch.pi * (frequency * time + common + mix * independent)
    amplitude = .5 * draw(batch, 1, 1).square() * wave.abs().square().mean(-1, keepdim=True).sqrt()
    tone = torch.polar(amplitude.expand_as(phase), phase)
    return noise + tone


@torch.no_grad()
def initialize(model, tracker, training, calibration):
    def framed(rows):
        return torch.cat([coordinates(tracker, batch)[0] for batch in rows.split(128)])

    clean = framed(calibration)
    balanced = training.unflatten(0, (len(SOURCE), -1))[:, :1000].flatten(0, 1)
    fit = framed(balanced)
    fitting = pair_spectrum(fit)
    sealed = pair_spectrum(clean)
    model.training_gate.copy_(clean_threshold(tracker, calibration))
    model.calibrated_noise.copy_(model.estimate_power(model.transformed(clean)).median())
    for name in ('near', 'middle'):
        fit_y, fit_w = band_observations(fitting, getattr(model, f'{name}_bins'))
        y_center, w_center = fit_y.mean((0, 1)), fit_w.mean((0, 1))
        getattr(model, f'{name}_center').copy_(y_center)
        getattr(model, f'{name}_reference').copy_(w_center)
        cross = ((fit_w - w_center).conj() * (fit_y - y_center).unsqueeze(-1)).mean((0, 1, 3))
        direction = cross / cross.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        getattr(model, f'{name}_blocked').copy_(direction.unsqueeze(-2))
        signal, reference = model.signed_residual(sealed, name)
        variation = signal.abs().square().mean((1, 3))
        groups = variation.unflatten(0, (len(SOURCE), -1))
        bound = 1.10 * torch.quantile(groups, .99, dim=1).amax(0)
        getattr(model, f'{name}_budget').copy_(bound)
        leakage = reference.abs().square().mean((0, 1, 3, 4))
        getattr(model, f'{name}_clean_leakage').copy_(leakage)
    rng = torch.Generator(device=clean.device).manual_seed(2026112410)
    views = [pair_spectrum(framed(real_iq(generic_observation(complex_iq(calibration), rng))))
             for _ in range(2)]
    for name in ('near', 'middle'):
        first, second = [model.band_prediction(view, name)[2] for view in views]
        difference = (first - second).abs().flatten(0, 1)
        getattr(model, f'{name}_margin').copy_(.25 * torch.quantile(difference, .50, dim=0))


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean = complex_iq(iq)
    framed, rotation, _, level = coordinates(tracker, iq)
    clean_frame = rotation * level
    weak_common = weak_difference(framed)
    change = weak_common * clean_frame
    wave = paired_inputs(clean, recipe.perturb, rng, change, with_identity=with_identity)
    observed, rotation, quality, level = coordinates(tracker, real_iq(wave.flatten(0, 1)))
    basis = rotation * level
    target = torch.stack((clean, clean, clean + change), dim=1).flatten(0, 1) / basis
    coarse, fine, prior, variance, diagnostics = model.read(observed, quality, level)
    error = (model.transformed(target) - prior).abs().square()
    likelihood = (error / variance + variance.log())[..., model.noncarrier].mean()
    physical = (fine * basis).unflatten(0, wave.shape[:2])
    consistent = ((physical[:, 0] - physical[:, 1]) / clean_frame).abs().square().mean()
    transported = (physical[:, 2] - physical[:, 0]) / clean_frame
    def split(value):
        common = value.mean(1, keepdim=True)
        return common, value - common
    passage = sum(transfer_risk(x, y) for x, y in zip(split(transported), split(weak_common), strict=True)) / 2
    fidelity = (model.transformed(observed - fine).abs().square() / (.025 + variance)).mean()
    values = torch.stack((waveform_risk(coarse, target), waveform_risk(fine, target), consistent,
                          passage, likelihood, fidelity, diagnostics[0].mean()))
    first, second, pair = recipe.loss_weights(step)
    ramp = min(1., step / 8000.)
    return (first * values[0] + second * values[1] + pair * consistent + .35 * ramp * passage
            + .08 * likelihood + .02 * ramp * fidelity), values
