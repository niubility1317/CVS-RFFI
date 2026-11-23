"""Release and score the matched LoRa cross-fitted cancellation arms."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze
from .reference_cancel import ReferenceCancel, generic_observation, initialize, objective, read_relative_action
from .spectral_block_source import load_models, main


RUN = '20261124-phase1-lora-reference-cancel-source-m4-r01'
RECIPES = {
    name: FitRecipe(name, 40_000, partial(ReferenceCancel, guarded=name == 'guarded'),
                    generic_observation, None, coarse_first=True, batch=24, fit_per_rx=4000,
                    validation_steps=(1800, 6000, 12000, 20000, 30000), initialize=initialize)
    for name in ('fixed', 'guarded')
}
MODES = ('direct', 'reference', 'sync', 'sync_wiener', 'fixed_coarse', 'fixed_full', 'fixed_half',
         'guarded_coarse', 'guarded_full', 'guarded_half', 'guarded_without_cancel',
         'guarded_without_action', 'guarded_without_noise', 'guarded_learned')
COMPARISONS = (('sync', 'direct'), ('sync_wiener', 'sync'),
               *((f'guarded_{kind}', f'fixed_{kind}') for kind in ('coarse', 'full', 'half')),
               ('guarded_full', 'guarded_coarse'),
               *(('guarded_full', mode) for mode in MODES[10:]),
               *((mode, 'reference') for mode in MODES[2:]))
ESTIMATES = ('active', 'near_common_dose', 'middle_common_dose', 'near_private_dose', 'middle_private_dose',
             'relative_action_confidence', 'relative_action_residual', 'outer_variance',
             'cancellation_rms', 'total_rms_fraction')


def recovery(iq, tracker, models, threshold, *, controls=False, summaries=False):
    y, rotation, quality, level = coordinates(tracker, iq)
    enabled = corruption_score(y) > threshold
    proposals = [iq.clone() for _ in range(3 * len(models) + (6 if controls else 0))]
    summary = iq.new_zeros((len(iq), len(ESTIMATES))) if summaries else None
    active = int(enabled.sum())
    if not active:
        return proposals, active, summary
    raw = iq[enabled]
    y, rotation, quality, level = (block[enabled] for block in (y, rotation, quality, level))
    basis = rotation * level
    updates = []
    for model in models:
        coarse, full, _, _, stats = model.read(y, quality, level)
        edit = real_iq((full - y) * basis)
        updates.extend((raw + real_iq((coarse - y) * basis), raw + edit, raw + .5 * edit))
    if summaries:
        dose, confidence, disagreement, cancellation_rms = stats
        variance = models[-1].estimate_power(models[-1].transformed(y)).mean((1, 2, 3))
        rms_fraction = (edit.square().sum((1, 2, 3)) / raw.square().sum((1, 2, 3)).clamp_min(1e-8)).sqrt()
        summary[enabled] = torch.stack((torch.ones_like(variance), *dose.mean(1).flatten(1).unbind(1),
                                       confidence.mean((1, 2)), disagreement.mean((1, 2)), variance,
                                       cancellation_rms, rms_fraction), dim=1)
    if controls:
        synced_frame = rotation * tracker.reference.chirp.conj()
        synced_frame = synced_frame[..., :1] * tracker.reference.chirp * level
        action, action_conf, _ = read_relative_action(y, quality)
        inverse = 1.05 * action.conj() / (.05 + action.abs().square())
        gain = 1. + action_conf * (inverse - 1.)
        equalized = y * gain.repeat_interleave(32, dim=-1)
        spectrum = torch.fft.fft(equalized, norm='ortho')
        estimated = models[-1].estimate_power(models[-1].transformed(y)).mean(2) * 2.
        shrink = (1. - estimated / (.02 + spectrum.abs().square())).clamp_min(0.)
        # Preserve the known carrier in the receive coordinates for the Wiener control.
        spectrum = torch.cat((torch.fft.fft(y, norm='ortho')[..., :1],
                              (spectrum * shrink)[..., 1:]), dim=-1)
        engineering = torch.fft.ifft(spectrum, norm='ortho')
        updates = [real_iq(y * synced_frame), real_iq(engineering * synced_frame), *updates]
        for action_on, noise_on, cancel_on in ((True, True, False), (False, True, True),
                                               (True, False, True), (False, False, False)):
            restored = models[-1].read(y, quality, level, use_action=action_on, use_noise=noise_on,
                                       use_cancel=cancel_on)[1]
            updates.append(raw + real_iq((restored - y) * basis))
    for output, proposed in zip(proposals, updates, strict=True):
        output[enabled] = proposed
    return proposals, active, summary


def render(iq, _unused, baseline, tracker, models, *, threshold, summaries=None):
    reference, _ = baseline(iq)
    proposals, active, diagnostics = recovery(iq, tracker, models, threshold, controls=True,
                                              summaries=summaries is not None)
    if summaries is not None:
        summaries.append(np.asarray(diagnostics.tolist()))
    return torch.stack((iq, reference, *proposals)), active


@torch.no_grad()
def smoke(rt, recipes, modes):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    (_, source, _), contract = setup(rt, device, smoke=True)
    models, threshold = load_models(root, contract, device, recipes)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if expected != contract or any(float(model.training_gate) != threshold for model in models):
        raise ValueError('The frozen receiver calibration must match the real source contract')
    reader = load_reader(root, contract, device)
    generator = torch.Generator(device=device).manual_seed(2026112411)
    candidates = real_iq(generic_observation(complex_iq(source), generator))
    quality = corruption_score(coordinates(tracker, candidates)[0])
    disturbed = candidates[quality.argmax().view(1)]
    incoming = torch.cat((source[:1], disturbed))
    outputs, active = render(incoming, None, baseline, tracker, models, threshold=threshold)
    logits = reader(outputs.flatten(0, 1))[0]
    if logits.shape != (2 * len(modes), 10) or not torch.isfinite(logits).all() or active != 1:
        raise ValueError('Expected paired real checkpoint outputs and clean bypass without target access')
    if not torch.equal(outputs[2:, 0], incoming[:1].expand_as(outputs[2:, 0])):
        raise ValueError('Clean bypass must preserve every sample')
    times = {}
    for name, sample in (('source_clean', source[:1]), ('generic_disturbed', disturbed)):
        values = []
        for step in range(28):
            torch.cuda.synchronize()
            begun = perf_counter()
            _, enabled, _ = recovery(sample, tracker, models[-1:], threshold)
            torch.cuda.synchronize()
            if step >= 8:
                values.append(1000 * (perf_counter() - begun))
        times[name] = dict(active=bool(enabled), median_ms=float(np.median(values)))
    calibration = {key: value.tolist() for key, value in models[-1].named_buffers()
                   if key.endswith(('_budget', '_margin', '_clean_leakage'))}
    write(root/'smoke.json', dict(status='PASS', V_reads=0, target_reads=0, device=torch.cuda.get_device_name(),
          b1=times, parameters={k:sum(p.numel() for p in m.parameters()) for k,m in zip(recipes,models,strict=True)},
          source_calibration=calibration))


def evaluate(rt, route, recipes, modes):
    device = 'cuda:0'
    binding = source_control(rt, route, device)
    models, threshold = load_models(Path(rt['run_root']), binding[2], device, recipes)
    summaries = []

    def finalize(root, route, source, chosen, ids, predictions, metrics, modes):
        data = np.concatenate(summaries).reshape(len(VIEWS), len(chosen), len(ESTIMATES))
        by_scene = {}
        for scene, records in zip(VIEWS, data, strict=True):
            by_scene[scene] = {}
            for rx, values in zip(SOURCE, np.split(records, len(SOURCE)), strict=True):
                active = values[values[:, 0] > 0]
                stats = {'active': len(active)}
                if len(active):
                    quantiles = np.quantile(active[:, 1:], (.1, .5, .9), axis=0).T.tolist()
                    stats.update(zip(ESTIMATES[1:], quantiles, strict=True))
                by_scene[scene][rx] = stats
        write(root/f'{route}-estimates.json', dict(quantiles=[.1, .5, .9], by_scene=by_scene))
        freeze(root, route, source, chosen, ids, predictions, metrics, modes)

    score(rt, route, panel=panel, modes=modes, render=partial(render, threshold=threshold, summaries=summaries),
          binding=binding, frozen_models=models, require_clean_bypass=False,
          replay_fn=receive_only, finalize=finalize)


if __name__ == '__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=smoke, run_evaluate=evaluate)
