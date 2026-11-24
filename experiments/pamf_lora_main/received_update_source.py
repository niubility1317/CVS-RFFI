"""Run matched directional and scalar LoRa source fits and truth-last scoring."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import write

from .channel_blind import corruption_score
from .channel_blind_source import SEED, FitRecipe, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import source_control
from .local_wiener_source import freeze
from .received_update import ReceivedUpdate, objective, update_coordinates
from .reference_cancel import generic_observation
from .spectral_block_source import load_models, main
from .variance_transport import match_dose, physical_norm


RUN = '20261124-phase1-lora-received-update-source-m4-r01'
DEVICE = 'cuda:0'
FIELDS = ('active', 'action_edit', 'outer_power', 'held_residual', 'rms_edit',
          'matched_rms', 'common_amp', 'common_phase', 'private_amp', 'private_phase')


def calibrate(model, tracker, calibration):
    grouped = calibration.unflatten(0, (len(SOURCE), -1))
    for_risk, for_gate = (x.flatten(0, 1) for x in grouped.chunk(2, dim=1))
    coordinates = torch.cat([update_coordinates(tracker, x)[0] for x in for_gate.split(32)])
    threshold = float(1.05 * corruption_score(coordinates).max())
    source_spectrum = model.components(torch.fft.fft(coordinates, norm='ortho')).abs()
    for band_id, mask in enumerate(model.bands.bool()):
        samples = source_spectrum[..., mask].transpose(0, 1).flatten(1)
        floor = (.25 * samples.quantile(.05, dim=1)).clamp_min(1e-4)
        model.axis_floor[:, band_id].copy_(floor)
    rng = torch.Generator(device=calibration.device).manual_seed(SEED + 23)
    risk_folds = for_risk.unflatten(0, (len(SOURCE), -1)).chunk(2, dim=1)
    for stage, fold in enumerate(risk_folds):
        limits = []
        for source in complex_iq(fold.flatten(0, 1)).split(24):
            y, phase, quality, rms = update_coordinates(tracker, real_iq(generic_observation(source, rng)))
            references = source / (phase * rms)
            mean, uncertainty, projection, active, _ = model.states(y, quality, rms, reference=references)[2][stage]
            errors = ((mean - projection) / uncertainty).masked_fill(~active, 0.)
            limits.append(errors.flatten(1).amax(1))
        limits = torch.cat(limits)
        # The second fold sees the already calibrated first stage. Allocate
        # 2.5% marginal miss probability to each stage.
        rank = min(len(limits), (39 * (len(limits) + 1) + 39) // 40)
        kappa = limits.kthvalue(rank).values.clamp_min(0.)
        if not torch.isfinite(kappa):
            raise FloatingPointError('Nonfinite LoRa direction calibration')
        model.kappa[stage].copy_(kappa)
    return threshold, dict(kappa=model.kappa.tolist(), calibration_risk_packets=len(for_risk),
                           calibration_clean_packets=len(for_gate), axis_floor=model.axis_floor.tolist())


RECIPES = {
    name: FitRecipe(name, 40_000, partial(ReceivedUpdate, scalar=name == 'scalar'), generic_observation,
                    None, coarse_first=True, batch=24, fit_per_rx=4000,
                    validation_steps=(1800, 6000, 12000, 20000, 30000), finalize_calibration=calibrate)
    for name in ('scalar', 'directional')
}
MODES = ('direct', 'reference', 'scalar_coarse', 'scalar_final', 'directional_coarse', 'directional_final',
         'directional_matched', 'scalar_matched', 'sync', 'sync_mmse', 'learning',
         'identity_action', 'fixed_variance', 'common_anchor')
COMPARISONS = (('directional_matched', 'scalar_matched'), ('directional_final', 'scalar_final'),
               ('directional_final', 'directional_coarse'), ('sync_mmse', 'sync'),
               *((mode, base) for mode in MODES[2:] for base in ('reference', 'direct')))


def engineering(y, phase, rms, quality, tracker, action, sigma):
    # With only upchirps this controls relative effective frequency within a
    # packet. Absolute STO/CFO and SFO are not recovered from the crop.
    frequency = (phase[..., 1:] * phase[..., :-1].conj())
    template = tracker.reference.chirp
    frequency = (frequency * (template[1:].conj() * template[:-1])).mean(-1).angle()
    offset = (frequency.median(1, keepdim=True).values - frequency).clamp(-.03, .03)
    offset = torch.where(quality[..., 0] > .15, offset, 0.)
    t = torch.arange(y.shape[-1], device=y.device)
    sync = torch.polar(torch.ones_like(offset[..., None] * t), offset[..., None] * t)
    equalized = (action.conj() * y + .05 * y) / (action.abs().square() + .05)
    spectrum = torch.fft.fft(equalized, norm='ortho')
    power = spectrum.abs().square().mean(1, keepdim=True)
    denoise = (power - sigma).clamp_min(0.) / (power + .25 * sigma + 1e-7)
    denoise[..., 0] = 1.
    mmse = torch.fft.ifft(denoise * spectrum, norm='ortho')
    return ((sync - 1.) * y * phase * rms, (mmse * sync - y) * phase * rms)


def recover(iq, tracker, models, threshold, *, matrix=False):
    wave, rotation, quality, scale = update_coordinates(tracker, iq)
    enabled = corruption_score(wave) > threshold
    count = int(enabled.sum())
    outputs = iq.expand(len(MODES) - 2 if matrix else 2 * len(models), *iq.shape).clone()
    metrics = iq.new_zeros((len(iq), len(FIELDS)))
    if not count:
        return outputs, count, metrics
    y, phase, q, rms = (value[enabled] for value in (wave, rotation, quality, scale))
    estimates = [model.states(y, q, rms) for model in models]
    edits = [out - y for result in estimates for out in result[:2]]
    physical = [edit * phase * rms for edit in edits]
    if matrix:
        model = models[-1]
        control = model.states(y, q, rms, scalar=True)[1] - y
        full = edits[-1]
        matched = match_dose(full, control, rms)
        physical.extend(v * phase * rms for v in (matched, match_dose(control, full, rms)))
        action, sigma, held = estimates[-1][3]
        physical.extend(engineering(y, phase, rms, q, tracker, action, sigma))
        for options in (dict(learning_only=True), dict(identity=True), dict(fixed=True), dict(anchor_common=True)):
            revised = model.states(y, q, rms, **options)[1]
            physical.append((revised - y) * phase * rms)
        gains = estimates[-1][2][-1][-1].mean(2).flatten(1)
        rms_original = physical_norm(y, rms).flatten().clamp_min(1e-7)
        edited = [physical_norm(v, rms).flatten() / rms_original for v in (full, matched)]
        summary = [torch.ones_like(edited[0]), (action - 1.).abs().mean((1, 2)),
                   sigma.mean((1, 2)), held.mean((1, 2)), *edited]
        metrics[enabled] = torch.cat((torch.stack(summary, -1), gains), -1)
    increments = torch.stack(physical)
    outputs[:, enabled] = iq[enabled] + torch.stack((increments.real, increments.imag), dim=-2)
    return outputs, count, metrics


def render(iq, _unused, baseline, tracker, models, *, threshold, measurements=None):
    reference, _ = baseline(iq)
    outputs, active, observed = recover(iq, tracker, models, threshold, matrix=True)
    if measurements is not None:
        measurements.append(np.asarray(observed.tolist(), dtype=np.float64))
    return torch.cat((torch.stack((iq, reference)), outputs)), active


@torch.no_grad()
def smoke(rt, recipes, modes):
    root = Path(rt['run_root'])
    (_, source, _), contract = setup(rt, DEVICE, smoke=True)
    models, threshold = load_models(root, contract, DEVICE, recipes)
    _, _, expected, baseline, tracker = source_control(rt, 'full', DEVICE)
    if expected != contract:
        raise ValueError('Expected the existing LoRa source contract')
    reader = load_reader(root, contract, DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(SEED + 41)
    perturbed = real_iq(generic_observation(complex_iq(source), rng))
    active = perturbed[corruption_score(update_coordinates(tracker, perturbed)[0]).argmax().view(1)]
    clean = source[corruption_score(update_coordinates(tracker, source)[0]).argmin().view(1)]
    outputs, activated = render(torch.cat((clean, active)), None, baseline, tracker, models, threshold=threshold)
    logits = reader(outputs.flatten(0, 1))[0]
    if activated != 1 or logits.shape != (2 * len(modes), 10) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite predictions and one active received path')
    if not torch.equal(outputs[2:, 0], clean.expand_as(outputs[2:, 0])):
        raise ValueError('LoRa must copy every clean sample on the bypass')
    times = {}
    for kind, iq in (('clean', clean), ('generic_disturbed', active)):
        delay = []
        for _ in range(28):
            torch.cuda.synchronize()
            start = perf_counter()
            wave = recover(iq, tracker, models[-1:], threshold)[0][-1]
            torch.cuda.synchronize()
            restore = perf_counter() - start
            reader(wave)
            torch.cuda.synchronize()
            delay.append((restore * 1e3, (perf_counter() - start) * 1e3))
        times[kind] = dict(zip(('recovery', 'with_reader'), np.median(delay[8:], axis=0).tolist(), strict=True))
    write(root / 'smoke.json', dict(status='PASS', V_reads=0, target_reads=0, timing_b1_ms=times,
                                  params=sum(p.numel() for p in models[-1].parameters())))


def evaluate(rt, route, recipes, modes):
    binding = source_control(rt, route, DEVICE)
    models, threshold = load_models(Path(rt['run_root']), binding[2], DEVICE, recipes)
    records = []

    def finalize(root, route, source, chosen, ids, predictions, metrics, modes):
        values = np.concatenate(records).reshape(len(VIEWS), len(chosen), len(FIELDS))
        received = {}
        for view, rows in zip(VIEWS, values, strict=True):
            received[view] = {}
            for group, cells in zip(SOURCE, np.split(rows, len(SOURCE)), strict=True):
                active = cells[cells[:, 0] > 0]
                medians = dict(zip(FIELDS[1:], np.median(active[:, 1:], axis=0), strict=True)) if len(active) else {}
                received[view][group] = dict(active=len(active), **medians)
        write(root / f'{route}-estimates.json', received)
        freeze(root, route, source, chosen, ids, predictions, metrics, modes)

    score(rt, route, panel=panel, modes=modes, binding=binding, frozen_models=models, replay_fn=receive_only,
          render=partial(render, threshold=threshold, measurements=records), require_clean_bypass=False,
          finalize=finalize)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=smoke, run_evaluate=evaluate)
