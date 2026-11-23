"""Run the signed-versus-magnitude LoRa cross-fit source study."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .crossfit_choice import PROPOSALS, CrossfitChoice, decision, objective
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_wiener_source import freeze
from .spectral_block_recovery import varied_observation
from .spectral_block_source import load_models, main


RUN = '20261124-phase1-lora-crossfit-choice-source-m4-r01'
RECIPES = {arm: FitRecipe(arm, 1800, partial(CrossfitChoice, signed), varied_observation, None)
           for arm, signed in (('magnitude', False), ('signed', True))}
MODES = ('direct', 'reference', 'magnitude_hard', 'magnitude_soft', 'signed_hard', 'signed_soft', *PROPOSALS[1:])
COMPARISONS = (('signed_hard', 'magnitude_hard'), ('signed_soft', 'magnitude_soft'),
               *((mode, 'reference') for mode in MODES[2:]))


def recover(iq, tracker, models, threshold, *, fixed=False):
    observed, rotation, quality, rms = coordinates(tracker, iq)
    active = corruption_score(observed)>threshold
    counts = []
    waves = [iq.clone() for _ in range(2*len(models)+(len(PROPOSALS)-1 if fixed else 0))]
    if active.any():
        z, q, base = observed[active], quality[active], rotation[active]*rms[active]
        received = iq[active]
        readouts = []
        for model in models:
            candidates, gains = model(z, q)
            hard, soft, choice = decision(candidates, gains)
            readouts.extend((hard, soft))
            counts.append(torch.bincount(choice, minlength=len(PROPOSALS)))
        if fixed:
            readouts.extend(candidates.unbind(1)[1:])
        for out, readout in zip(waves, readouts, strict=True):
            out[active] = received+real_iq((readout-z)*base)
    return waves, int(active.sum()), counts


def render(iq, _unused, baseline, tracker, models, *, threshold, totals=None):
    reference, _ = baseline(iq)
    waves, active, choices = recover(iq, tracker, models, threshold, fixed=True)
    if totals is not None and active:
        totals.add_(torch.stack(choices))
    return torch.stack((iq, reference, *waves)), active


@torch.no_grad()
def smoke(rt, recipes, modes):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    (_, source, _), contract = setup(rt, device, smoke=True)
    models, threshold = load_models(root, contract, device, recipes)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('The calibration and reference must share the source contract')
    reader = load_reader(root, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026112329)
    received = real_iq(varied_observation(complex_iq(source), rng))
    ratios = corruption_score(coordinates(tracker, received)[0])
    impaired = received[ratios.argmax().view(1)]
    waves, active = render(torch.cat((source[:1], impaired)), None, baseline, tracker, models, threshold=threshold)
    logits = reader(waves.flatten(0,1))[0]
    if logits.shape != (2*len(modes),10) or not torch.isfinite(logits).all() or not active:
        raise ValueError('Expected real frozen Tri-set, recovery and DURA outputs')
    times = {}
    for name, sample in (('source_clean', source[:1]), ('generic_disturbed', impaired)):
        records = []
        for step in range(28):
            torch.cuda.synchronize()
            begun = perf_counter()
            _, enabled, _ = recover(sample, tracker, (models[1],), threshold)
            torch.cuda.synchronize()
            if step >= 8:
                records.append(1000*(perf_counter()-begun))
        times[name] = dict(active=bool(enabled), median_ms=float(np.median(records)))
    write(root/'smoke.json', dict(status='PASS', target_reads=0, V_reads=0,
          parameters={name:sum(p.numel() for p in model.parameters()) for name,model in zip(recipes,models,strict=True)},
          b1=times, device=torch.cuda.get_device_name()))


def evaluate(rt, route, recipes, modes):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    binding = source_control(rt, route, device)
    models, threshold = load_models(root, binding[2], device, recipes)
    choices = torch.zeros((len(recipes),len(PROPOSALS)), dtype=torch.long, device=device)
    score(rt, route, panel=panel, modes=modes, render=partial(render, threshold=threshold, totals=choices),
          binding=binding, frozen_models=models, require_clean_bypass=False, replay_fn=receive_only, finalize=freeze)
    write(root/f'{route}-choices.json', dict(proposals=list(PROPOSALS), active_by_arm=dict(zip(recipes,choices.tolist(),strict=True))))


if __name__=='__main__':
    main(run=RUN, runner=__spec__.name, recipes=RECIPES, modes=MODES, comparisons=COMPARISONS,
         calculate=objective, run_smoke=smoke, run_evaluate=evaluate)
