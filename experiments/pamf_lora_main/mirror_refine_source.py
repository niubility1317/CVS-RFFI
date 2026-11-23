"""Train and source-score LoRa refinement along trusted mirror directions."""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, fit, panel, proof, receive_only, setup
from .chirp_spectrum_source import dispatch, freeze, read_frozen, score_frozen
from .complex_vector_source import load_reader, publish, score
from .data import VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .mirror_refine import BANDS, MirrorRefine
from .natural_detail import natural_objective, paired_echo_inputs
from .rf_noise_transport import colored_observation
from .rf_noise_transport_source import timing
from .runner import activate
from .spectral_unroll import ROUTES
from .tail_pilot_source import MIRROR_MODES, MIRROR_RECIPES, MIRROR_RUN, STEPS, TAIL_WEIGHT, WIDTHS


RUN = '20261124-phase1-lora-mirror-protected-refine-source-m3-r01'
RECIPE = FitRecipe('protected', STEPS, partial(MirrorRefine, WIDTHS), colored_observation, MirrorRefine.forward)
MODES = ('direct', 'reference', 'mirror_coarse', 'mirror', 'mirror_half',
         'frozen_protected', 'frozen_protected_half', 'trained_coarse', 'protected', 'protected_half',
         'dose_only', 'dose_only_half')
PAIRS = (('frozen_protected', 'mirror'), ('frozen_protected_half', 'mirror_half'),
         ('trained_coarse', 'mirror_coarse'), ('protected', 'mirror'), ('protected_half', 'mirror_half'),
         ('protected', 'dose_only'), ('protected_half', 'dose_only_half'))
PARENT_READOUTS = MODES[:5]


def load_model(root, recipe, contract, device, *, model=None):
    stored = torch.load(root / f'{recipe.name}-model.pt', map_location=device, weights_only=True)
    if stored['proof'] != proof(contract, recipe) or not np.isfinite(stored['threshold']):
        raise ValueError(f'Invalid source-only checkpoint or threshold for {recipe.name}')
    model = recipe.model() if model is None else model
    model.to(device).eval().requires_grad_(False)
    model.load_state_dict(stored['weights'], strict=True)
    return model, stored['threshold']


def frozen_mirror(root, contract, device):
    parent = root.parent / MIRROR_RUN
    if read(parent / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected the completed source-only mirror checkpoint')
    return load_model(parent, MIRROR_RECIPES['mirror'], contract, device, model=MirrorRefine(WIDTHS))


@torch.no_grad()
def render(iq, _unused, baseline, tracker, models, *, threshold, support=None):
    reference, _ = baseline(iq)
    observed, basis, quality, rms = coordinates(tracker, iq)
    active = corruption_score(observed) > threshold
    if not bool(active.any()):
        return torch.stack((iq, reference, *([iq] * (len(MODES) - 2)))), 0
    y, rotation, q, scale = (value[active] for value in (observed, basis, quality, rms))
    parent, candidate = models
    coarse, ordinary, projected, _, _ = parent(y, q, rotation, controls=True)
    trained_coarse, _, limited, dose_only, confidence = candidate(y, q, rotation, controls=True)
    if support is not None:
        support[0] += confidence.sum(dim=(0, 1))
        support[1] += (confidence > 0).sum(dim=(0, 1))
        support[2] += confidence.shape[0] * confidence.shape[1]
    values = (coarse, ordinary, torch.lerp(y, ordinary, .5), projected, torch.lerp(y, projected, .5),
              trained_coarse, limited, torch.lerp(y, limited, .5), dose_only, torch.lerp(y, dose_only, .5))
    outputs = [iq, reference]
    for value in values:
        out = iq.clone()
        out[active] += real_iq((value - y) * rotation * scale)
        outputs.append(out)
    return torch.stack(outputs), int(active.sum())


@torch.no_grad()
def checkpoint_smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, calibration, _), contract = setup(rt, device, smoke=True)
    _, _, expected, baseline, tracker = source_control(rt, 'full', device)
    if expected != contract:
        raise ValueError('Mismatched real LoRa source U and frozen recognition contract')
    parent, threshold = frozen_mirror(root, contract, device)
    candidate = RECIPE.model().to(device).eval()
    rng = torch.Generator(device=device).manual_seed(2026112403)
    batch = real_iq(colored_observation(complex_iq(calibration[:2]), rng))
    waves, active = render(batch, None, baseline, tracker, (parent, candidate), threshold=-1.)
    logits = load_reader(root, contract, device)(waves.flatten(0, 1))[0]
    if active != len(batch) or logits.shape != (len(MODES) * len(batch), 10) or not torch.isfinite(logits).all():
        raise ValueError('Real no-query mirror refinement checkpoint smoke failed')
    write(root / 'checkpoint_smoke.json', dict(status='PASS', source_U=len(batch), variants=len(MODES),
                                              inherited_threshold=threshold, target_reads=0))
    print('LORA_MIRROR_REFINE_REAL_CHECKPOINT_PASS', flush=True)


def evaluate(rt, route):
    root, device = Path(rt['run_root']), 'cuda:0'
    binding = source_control(rt, route, device)
    parent, threshold = frozen_mirror(root, binding[2], device)
    candidate, candidate_threshold = load_model(root, RECIPE, binding[2], device)
    if threshold != candidate_threshold:
        raise ValueError('Candidate and parent must use the identical source U gate')
    support = torch.zeros((3, len(BANDS)), device=device)
    score(rt, route, panel=panel, modes=MODES, render=partial(render, threshold=threshold, support=support),
          binding=binding, frozen_models=(parent, candidate), replay_fn=receive_only,
          finalize=freeze, reader_chunk=144)
    normalizer = support[2].clamp_min(1)
    write(root / f'{route}-support.json', dict(bands_khz=BANDS, active_chirps=int(support[2, 0]),
              mean_confidence=(support[0] / normalizer).tolist(),
              support_fraction=(support[1] / normalizer).tolist()))


def independent_score(rt):
    root = Path(rt['run_root'])
    for route in ROUTES:
        ids, current, _ = read_frozen(root, route, MODES)
        previous_ids, previous, _ = read_frozen(root.parent / MIRROR_RUN, route, MIRROR_MODES)
        if ids != previous_ids:
            raise ValueError('Frequency refinement panel differs from the frozen mirror parent')
        for view in VIEWS:
            for arm in PARENT_READOUTS:
                if not np.array_equal(current[view][MODES.index(arm)], previous[view][MIRROR_MODES.index(arm)]):
                    raise ValueError(f'Frozen parent predictions differ on {route}/{view}/{arm}')
    score_frozen(rt, modes=MODES, comparisons=PAIRS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for action in ('dispatch', 'checkpoint_smoke', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        cmd = commands.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        if action != 'dispatch':
            names = ROUTES if action == 'evaluate' else ('protected',) if action in ('smoke', 'fit') else ('source',)
            cmd.add_argument('name', choices=names)
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'publish':
        publish(**options, run=RUN, runner=__spec__.name)
        return
    rt = read(options['runtime'])
    if action == 'dispatch':
        dispatch(rt, (RECIPE.name,), initial_stages=(('checkpoint_smoke', ('source',)),))
    elif action == 'score':
        independent_score(rt)
    else:
        activate(rt)
        if action == 'checkpoint_smoke':
            checkpoint_smoke(rt)
        elif action == 'evaluate':
            evaluate(rt, options['name'])
        elif action == 'timing':
            timing(rt, RECIPE)
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPE, receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_echo_inputs,
                                  tail_weight=TAIL_WEIGHT))


if __name__ == '__main__':
    main()
