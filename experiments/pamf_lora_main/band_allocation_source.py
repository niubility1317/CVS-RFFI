"""Train fixed-dose LoRa band allocation using only received and legal source IQ."""

from functools import partial

import numpy as np
import torch

from .band_allocation import BandAllocation, apply_gain, band_energy, equal_budget, fft, ifft
from .channel_blind import corruption_score
from .channel_blind_source import SEED, FitRecipe, clean_threshold
from .data import SOURCE
from .dura_repair import complex_iq, real_iq
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, weak_difference
from .phase_dose_bursts_source import burst_observation
from .received_update_source import engineering
from .spectral_block_source import main


RUN = '20261124-phase1-lora-band-allocation-source-m3-r01'
MODES = ('direct', 'reference', 'sync', 'sync_mmse', 'semi_blind', 'anchor', 'observed_allocation',
         'learned_coarse', 'learned_final', 'same_weight_equal', 'no_feedback', 'unit_action', 'fixed_noise')
COMPARISONS = (('learned_final', 'same_weight_equal'), ('learned_final', 'observed_allocation'),
               ('learned_final', 'learned_coarse'), ('learned_final', 'no_feedback'),
               ('learned_final', 'unit_action'), ('learned_final', 'fixed_noise'),
               ('observed_allocation', 'anchor'), ('sync_mmse', 'sync'),
               *((name, control) for name in MODES[2:] for control in ('direct', 'reference')))
FIELDS = ('active', 'common_quadrature_variance', 'private_quadrature_variance', 'held_residual',
          'action_mean_change', 'base_near_share', 'base_mid_share', 'base_far_share', 'near_gain',
          'mid_gain', 'far_gain', 'observed_near_gain', 'observed_mid_gain', 'observed_far_gain',
          'anchor_rms_fraction', 'learned_rms_fraction', 'observed_rms_fraction',
          'anchor_to_learned_rms_error', 'anchor_to_observed_rms_error')


def calibration_roles(calibration):
    by_receiver = calibration.unflatten(0, (len(SOURCE), -1))
    gate, first, last = by_receiver.split((by_receiver.shape[1] // 2,
                                          by_receiver.shape[1] // 4,
                                          by_receiver.shape[1] // 4), dim=1)
    return tuple(group.flatten(0, 1) for group in (gate, first, last))


@torch.no_grad()
def initialize(model, tracker, training, calibration):
    model.learn_source_statistics(coordinates(tracker, batch)[0] for batch in training.split(96))
    gate, _, _ = calibration_roles(calibration)
    model.clean_level.copy_(clean_threshold(tracker, gate))


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean = complex_iq(iq)
    framed, angle, _, clean_rms = coordinates(tracker, iq)
    clean_frame = angle * clean_rms
    natural_change = weak_difference(framed)
    change = natural_change * clean_frame
    wave = paired_inputs(clean, recipe.perturb, rng, change, with_identity=with_identity)
    y, angle, quality, rms = coordinates(tracker, real_iq(wave.flatten(0, 1)))
    received_frame = angle * rms
    reference = torch.stack((clean, clean, clean + change), dim=1).flatten(0, 1) / received_frame
    state = model.observe(y, quality, rms)
    coarse, fine, _, risk = model.stages(y, rms, state)
    live = eligible(y, quality, model.clean_level)
    active = live.reshape(-1, 1, 1)
    proposed_fine = fine
    coarse, fine = coarse * active, fine * active
    y0, y1 = y + coarse, y + fine
    first = model.wave_risk(y0 - reference, reference)
    final = model.wave_risk(y1 - reference, reference)
    target = model.target_projection(reference - y, rms, state).clamp(-10., 10.)
    usable = state[3] & active.view(-1, 1)
    scoring = torch.stack([((((estimate-target).abs() / scale) + scale.log()) * usable).sum()
                           / usable.sum().clamp_min(1) for estimate, scale in risk]).mean()
    forward = fft(y - state[5] * y1)
    measured = band_energy(forward, 1., model.bands) / (6. * model.bands.sum(-1))
    sigma2 = 2. * state[6].mean(-1, keepdim=True)
    uncertainty = sigma2 + state[7].mean((1, 2))[:, None] + .01
    closure = (((measured / uncertainty) - 1.).clamp_min(0.).square() * active.view(-1, 1)).mean()
    restored = (y1 * received_frame).unflatten(0, wave.shape[:2])
    repeat = (((restored[:, 0] - restored[:, 1]) / clean_frame).abs().square()).mean()
    transferred = (restored[:, 2] - restored[:, 0]) / clean_frame
    preserve = model.difference_risk(transferred, natural_change)
    identity_input = torch.zeros_like(live.reshape(-1, 3))
    if with_identity:
        identity_input[-2:, :2] = True
    identity = (proposed_fine.abs().square().mean((1, 2)) * identity_input.flatten()).mean()
    first_weight, final_weight, repeat_weight = recipe.loss_weights(step)
    values = torch.stack((first, final, scoring, closure, repeat, preserve, identity))
    weights = values.new_tensor((first_weight, final_weight, .06, .025, repeat_weight, .35, .2))
    return values @ weights, values


@torch.no_grad()
def calibrate(model, tracker, calibration):
    gate, coarse_fold, fine_fold = calibration_roles(calibration)
    threshold = float(model.clean_level)
    rng = torch.Generator(device=calibration.device).manual_seed(SEED + 37)
    estimates = []
    covered_counts = []
    for stage, examples in enumerate((coarse_fold, fine_fold)):
        errors = []
        for raw in examples.split(24):
            source = complex_iq(raw)
            wave = burst_observation(source, rng)
            y, angle, quality, rms = coordinates(tracker, real_iq(wave))
            state = model.observe(y, quality, rms)
            if stage == 0:
                estimates.append(state[6])
            _, _, _, risk = model.stages(y, rms, state)
            estimate, scale = risk[stage]
            expected = model.target_projection(source / (angle*rms) - y, rms, state)
            usable = state[3] & eligible(y, quality, threshold).reshape(-1, 1)
            discrepancy = (estimate - expected) / scale
            errors.append(discrepancy.masked_fill(~usable, -torch.inf).amax(-1)[usable.any(-1)])
        errors = torch.cat(errors)
        # A one-step run uses only preregistered no-query smoke inputs.
        if errors.numel():
            rank = min(len(errors), (39 * (len(errors) + 1) + 39) // 40)
            model.margins[stage].copy_(errors.kthvalue(rank).values.clamp_min(0))
        covered_counts.append(int(errors.numel()))
    model.noise_baseline.copy_(torch.cat(estimates).median(0).values.clamp_min(1e-8))
    diagnostics = dict(calibration_margins=model.margins.tolist(), gate_packets=len(gate),
                       risk_packets=[len(coarse_fold), len(fine_fold)], active_risk_packets=covered_counts,
                       source_perturbed_noise_baseline=model.noise_baseline.tolist(),
                       prior_floats=sum(p.numel() for p in model.prior.state_dict().values()))
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite source-only band allocation calibration')
    return threshold, diagnostics


RECIPES = {'band': FitRecipe('band', 40_000, BandAllocation, burst_observation, None,
                            coarse_first=True, batch=24, fit_per_rx=4000, initialize=initialize,
                            validation_steps=(1800, 6000, 12000, 20000, 30000), finalize_calibration=calibrate)}


def eligible(y, quality, threshold):
    return (corruption_score(y) > threshold) & (quality.amax((1, 2)) > .15)


def recover(iq, tracker, models, threshold, *, matrix=False):
    model, = models
    y, angle, quality, rms = coordinates(tracker, iq)
    gate = eligible(y, quality, threshold)
    count = int(gate.sum())
    outputs = iq.expand(len(MODES) - 2 if matrix else 1, *iq.shape).clone()
    records = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return outputs, count, records
    y, angle, quality, rms = (value[gate] for value in (y, angle, quality, rms))
    state = model.observe(y, quality, rms)
    coarse, fine, gains, _ = model.stages(y, rms, state)
    frame = angle * rms
    if not matrix:
        update = real_iq(fine * frame)
        update = torch.where(torch.isfinite(update).all((1, 2, 3))[:, None, None, None], update, 0.)
        outputs[0, gate] = iq[gate] + update
        return outputs, count, records

    direction, energy, estimate, usable, _, action, noise, mismatch, shares = state
    observed_gains = equal_budget(energy, estimate, usable)
    observed = ifft(apply_gain(direction, observed_gains, model.bands))
    anchor = ifft(direction)
    sync, sync_mmse = engineering(y, angle, rms, quality, tracker, action,
                                  (2 * noise.mean(-1)).reshape(-1, 1, 1))
    equalized = (action.conj() * y + .05 * y) / (action.abs().square() + .05)
    semi_blind = equalized - y
    semi_blind = semi_blind - semi_blind.mean(-1, keepdim=True)
    no_feedback = model.stages(y, rms, state, use_residual=False)[1]
    no_action = model.stages(y, rms, model.observe(y, quality, rms, action_enabled=False))[1]
    no_adaptive_noise = model.stages(y, rms, model.observe(y, quality, rms, adaptive_noise=False))[1]
    edits = torch.stack((sync, sync_mmse, semi_blind*frame, anchor*frame, observed*frame, coarse*frame,
                         fine*frame, anchor*frame, no_feedback*frame, no_action*frame, no_adaptive_noise*frame))
    outputs[:, gate] = iq[gate] + torch.stack((edits.real, edits.imag), dim=-2)
    norms = edits[[3, 6, 4]].abs().square().mean((2, 3)).sqrt().T
    input_rms = (y*frame).abs().square().mean((1, 2)).sqrt().clamp_min(1e-8)
    errors = (norms[:, 1:] - norms[:, :1]).abs()
    if not torch.isfinite(edits).all() or (errors > (5e-6 * (1. + input_rms[:, None]))).any():
        raise ValueError('LoRa allocation must preserve real IQ RMS')
    records[gate] = torch.cat((torch.ones_like(noise[:, :1]), noise, mismatch.mean((1, 2))[:, None],
                               (action-1).abs().mean((1, 2))[:, None], shares, gains, observed_gains,
                               norms / input_rms[:, None], errors), dim=1)
    return outputs, count, records


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None, switch=None):
    reference, _ = baseline(iq)
    output, count, records = recover(iq, tracker, models, threshold, matrix=True)
    if measurements is not None:
        measurements.append(np.asarray(records.tolist()))
    return torch.cat((torch.stack((iq, reference)), output)), count


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=partial(smoke, render_fn=render, recover_fn=recover),
         run_evaluate=partial(evaluate, fields=FIELDS, render_fn=render))
