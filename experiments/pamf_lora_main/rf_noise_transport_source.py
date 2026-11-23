"""Train and test matched LoRa RF noise transport using observed IQ only."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, proof, setup
from .chirp_spectrum_source import dispatch, evaluate, score_frozen
from .complex_vector_source import load_reader, publish
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .natural_detail import natural_objective, paired_echo_inputs
from .rf_noise_confidence import RFNoiseConfidence
from .rf_noise_transport import RFNoiseTransport, colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-rf-noise-transport-source-m4-r01'
CONFIDENCE_RUN = '20261124-phase1-lora-rf-noise-confidence-source-m4-r01'
STEPS = 1800
MODES = ('direct', 'reference', 'flat_coarse', 'flat', 'flat_half',
         'transport_coarse', 'transport', 'transport_half')
PAIRS = (('transport_coarse', 'flat_coarse'), ('transport', 'flat'), ('transport_half', 'flat_half'))
RECIPES = {name: FitRecipe(name, STEPS, partial(RFNoiseTransport, aligned), colored_observation, RFNoiseTransport.forward)
           for name, aligned in (('flat', False), ('transport', True))}
CONFIDENCE_MODES = ('direct', 'reference', 'kernel_coarse', 'kernel', 'kernel_half',
                    'confidence_coarse', 'confidence', 'confidence_half')
CONFIDENCE_PAIRS = (('confidence_coarse', 'kernel_coarse'), ('confidence', 'kernel'),
                    ('confidence_half', 'kernel_half'))
CONFIDENCE_RECIPES = {
    name: FitRecipe(name, STEPS, partial(RFNoiseConfidence, adaptive), colored_observation, RFNoiseTransport.forward)
    for name, adaptive in (('kernel', False), ('confidence', True))
}


@torch.no_grad()
def proposals(iq, tracker, models, threshold):
    y, rotation, q, rms = coordinates(tracker, iq)
    active = corruption_score(y) > threshold
    if not bool(active.any()):
        return [iq] * (3 * len(models)), 0
    y, rotation, q, rms = (x[active] for x in (y, rotation, q, rms))
    result = []
    for model in models:
        coarse, fine = model(y, q, rotation)
        for value in (coarse, fine, torch.lerp(y, fine, .5)):
            output = iq.clone()
            output[active] += real_iq((value - y) * rotation * rms)
            result.append(output)
    return result, int(active.sum())


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    alternatives, active = proposals(iq, tracker, models, threshold)
    return torch.stack((iq, reference, *alternatives)), active


@torch.no_grad()
def checkpoint_smoke(rt, recipes, modes):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if contract != expected:
        raise ValueError('Source U does not match the real LoRa checkpoint contract')
    reader = load_reader(root, contract, device)
    models = [recipe.model().to(device).eval() for recipe in recipes.values()]
    generator = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), generator))
    values, _ = render(batch, None, baseline, tracker, models, threshold=-1.)
    logits = reader(values.flatten(0, 1))[0]
    if logits.shape != (len(modes) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('LoRa current IQ and real-reader smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', source_U=len(batch), target_reads=0))
    print('LORA_RF_NOISE_REAL_CHECKPOINT_PASS', flush=True)


@torch.no_grad()
def timing(rt, recipe):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    tracker = PacketRestorer('consensus').to(device).eval()
    saved = torch.load(root / f'{recipe.name}-model.pt', map_location=device, weights_only=True)
    if saved['proof'] != proof(contract, recipe):
        raise ValueError('Cannot time another LoRa restore checkpoint')
    model = recipe.model().to(device).eval()
    model.load_state_dict(saved['weights'], strict=True)
    generator = torch.Generator(device=device).manual_seed(2026112403)
    disturbed = real_iq(colored_observation(complex_iq(calibration), generator))
    case = corruption_score(coordinates(tracker, disturbed)[0]).argmax().view(1)
    report = {}
    for label, iq in (('source_clean', calibration[:1]), ('generic_disturbed', disturbed[case])):
        latency = []
        for i in range(32):
            torch.cuda.synchronize()
            start = perf_counter()
            _, active = proposals(iq, tracker, (model,), saved['threshold'])
            torch.cuda.synchronize()
            if i >= 8:
                latency.append(1000 * (perf_counter() - start))
        report[label] = dict(active=bool(active), median_ms=float(np.median(latency)))
    write(root / 'b1.json', dict(parameters=sum(p.numel() for p in model.parameters()),
                                 timing=report, gpu=torch.cuda.get_device_name(),
                                 n_measurements=24, torch=torch.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + key, type=Path, required=True)
    pub.add_argument('--confidence-study', action='store_true')
    fit_arms = (*RECIPES, *CONFIDENCE_RECIPES)
    for action in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        cmd = commands.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        if action != 'dispatch':
            cmd.add_argument('name', choices=fit_arms if action in ('smoke', 'fit') else
                             ROUTES if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        run = CONFIDENCE_RUN if args.pop('confidence_study') else RUN
        publish(**args, run=run, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    run = Path(rt['run_root']).name
    if run == CONFIDENCE_RUN:
        recipes, modes, pairs = CONFIDENCE_RECIPES, CONFIDENCE_MODES, CONFIDENCE_PAIRS
        measured = 'confidence'
    elif run == RUN:
        recipes, modes, pairs = RECIPES, MODES, PAIRS
        measured = 'transport'
    else:
        raise ValueError('Expected a registered LoRa RF noise estimation run')
    if action == 'dispatch':
        dispatch(rt, recipes, initial_stages=(('checkpoint_smoke', ('source',)),))
    elif action == 'score':
        score_frozen(rt, modes=modes, comparisons=pairs)
    else:
        activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt, recipes, modes)
        elif action == 'evaluate':
            evaluate(rt, args['name'], recipes=recipes, modes=modes, renderer=render)
        elif action == 'timing':
            timing(rt, recipes[measured])
        else:
            fit(rt, smoke=action == 'smoke', recipe=recipes[args['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_echo_inputs))


if __name__ == '__main__':
    main()
