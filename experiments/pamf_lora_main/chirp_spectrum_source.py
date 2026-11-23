"""Compare common and separate spectral shrinkage on matched LoRa source packets."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write

from .channel_blind import corruption_score, generic_observation
from .channel_blind_source import FitRecipe, fit, panel, proof, receive_only, setup
from .chirp_spectrum_recovery import ChirpSpectrumRecovery
from .complex_vector_source import load_reader, publish, score
from .data import SOURCE, VIEWS
from .denoise_source import EVAL_PER_RX, summarize
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_contract
from .joint_unroll_source import coordinates, source_control
from .natural_detail import natural_objective, paired_echo_inputs
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261123-phase1-lora-chirp-spectrum-recovery-source-m4-r01'
STEPS = 1200
MODES = ('direct', 'reference', 'shared', 'separate', 'separate_half')


def restore(model, observed, quality, _rotation):
    return model(observed, quality)


RECIPES = {
    arm: FitRecipe(arm, STEPS, partial(ChirpSpectrumRecovery, separate), generic_observation, restore)
    for arm, separate in (('shared', False), ('separate', True))
}


@torch.no_grad()
def recover(iq, tracker, models, threshold, half):
    observed, rotation, quality, rms = coordinates(tracker, iq)
    active = corruption_score(observed) > threshold
    outputs = []
    if not bool(active.any()):
        return [iq] * (len(models) + int(half)), 0
    y, q, basis = observed[active], quality[active], rotation[active] * rms[active]
    for model in models:
        _, refined = model(y, q)
        delta = real_iq((refined - y) * basis)
        full = iq.clone()
        full[active] = iq[active] + delta
        outputs.append(full)
    if half:
        midpoint = iq.clone()
        midpoint[active] = iq[active] + .5 * delta
        outputs.append(midpoint)
    return outputs, int(active.sum())


def render(iq, _unused, baseline, tracker, models, *, threshold):
    reference, _ = baseline(iq)
    values, active = recover(iq, tracker, models, threshold, half=True)
    return torch.stack((iq, reference, *values)), active


def freeze(root, route, _source, chosen, ids, predictions, metrics, modes):
    write(root / f'{route}-predictions.json', dict(ids=ids[chosen].tolist(),
          rows={k: v.tolist() for k, v in predictions.items()}, metrics=metrics, modes=list(modes)))


def evaluate(rt, route, recipes=RECIPES, modes=MODES, renderer=render):
    root = Path(rt['run_root'])
    device = 'cuda:0'
    binding = source_control(rt, route, device)
    models, thresholds = [], []
    for arm, recipe in recipes.items():
        saved = torch.load(root / f'{arm}-model.pt', map_location=device, weights_only=True)
        if saved['proof'] != proof(binding[2], recipe):
            raise ValueError('Unexpected source restoration checkpoint')
        model = recipe.model().to(device).eval().requires_grad_(False)
        model.load_state_dict(saved['weights'], strict=True)
        models.append(model)
        thresholds.append(saved['threshold'])
    if thresholds[0] != thresholds[1] or not np.isfinite(thresholds[0]):
        raise ValueError('Both restoration arms require the same source U threshold')
    score(rt, route, panel=panel, modes=modes, render=partial(renderer, threshold=thresholds[0]),
          binding=binding, frozen_models=models, require_clean_bypass=False, replay_fn=receive_only,
          finalize=freeze)


def read_frozen(root, route, modes):
    stored = read(root / f'{route}-predictions.json')
    count = len(SOURCE) * EVAL_PER_RX
    ids = stored['ids']
    rows = {view: np.asarray(value) for view, value in stored['rows'].items()}
    if (len(ids) != count or len(set(ids)) != count or stored['modes'] != list(modes)
            or set(rows) != set(VIEWS) or set(stored['metrics']) != set(VIEWS)
            or any(value.shape != (len(modes), count) or value.dtype.kind not in 'iu'
                   or (value < 0).any() or (value >= 10).any() for value in rows.values())):
        raise ValueError('Incomplete source V prediction matrix')
    return ids, rows, stored['metrics']


def score_frozen(rt, modes=MODES, comparisons=(('separate', 'shared'), ('separate_half', 'direct'))):
    root = Path(rt['run_root'])
    source = Path(rt['source_root'])
    frozen = {route: read_frozen(root, route, modes) for route in ROUTES}
    if frozen['full'][0] != frozen['residual'][0]:
        raise ValueError('Source prediction packets differ between routes')
    ids = np.asarray(source_contract(source)['role_ids']['V'])
    positions = {identity: i for i, identity in enumerate(ids)}
    chosen = np.asarray([positions[i] for i in frozen['full'][0]])
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as labels:
        if not np.array_equal(panel(np.asarray(SOURCE)[labels['rx_i']]), chosen):
            raise ValueError('Frozen packets differ from the registered source V selection')
    for route, (_, rows, metrics) in frozen.items():
        summarize(root, route, source, chosen, ids, rows, metrics, modes, predictions_written=True,
                  comparisons=comparisons)
    print('LORA_RESTORATION_SOURCE_SCORED', flush=True)


@torch.no_grad()
def measure(rt, recipe=RECIPES['separate'], *, verify_reader=True):
    root = Path(rt['run_root'])
    (_, calibration, _), contract = setup(rt, 'cuda:0', smoke=True)
    saved = torch.load(root / f'{recipe.name}-model.pt', map_location='cuda:0', weights_only=True)
    if saved['proof'] != proof(contract, recipe):
        raise ValueError('Cannot time a different restoration')
    model = recipe.model().cuda().eval()
    model.load_state_dict(saved['weights'], strict=True)
    tracker = PacketRestorer('consensus').cuda().eval()
    rng = torch.Generator(device='cuda:0').manual_seed(2026112302)
    clean = calibration[:1]
    wave = real_iq(generic_observation(complex_iq(calibration), rng))
    power_proxy = corruption_score(coordinates(tracker, wave)[0])
    corrupt = wave[power_proxy.argmax().view(1)]
    if verify_reader:
        _, _, bound_contract, baseline, _ = source_control(rt, 'full', 'cuda:0')
        if bound_contract != contract:
            raise ValueError('Source U timing and frozen reader contracts differ')
        reader = load_reader(root, contract, 'cuda:0')
        proposal, _ = recover(corrupt, tracker, (model,), saved['threshold'], half=False)
        reference, _ = baseline(corrupt)
        logits = reader(torch.cat((corrupt, reference, *proposal)))[0]
        if logits.shape != (3, 10) or not torch.isfinite(logits).all():
            raise ValueError('Real frozen LoRa checkpoint smoke failed')
        print('LORA_RESTORATION_REAL_CHECKPOINT_PASS', flush=True)
    results = {}
    for name, value in (('source_clean', clean), ('generic_disturbed', corrupt)):
        start_ms = []
        for iteration in range(32):
            torch.cuda.synchronize()
            begin = perf_counter()
            _, active = recover(value, tracker, (model,), saved['threshold'], half=False)
            torch.cuda.synchronize()
            if iteration >= 8:
                start_ms.append((perf_counter() - begin) * 1000)
        results[name] = dict(active=bool(active), median_ms=float(np.median(start_ms)))
    write(root / 'b1.json', dict(parameters=sum(p.numel() for p in model.parameters()),
          timing=results, torch=torch.__version__, gpu=torch.cuda.get_device_name()))


def dispatch(rt, recipes=RECIPES, *, initial_stages=()):
    root = Path(rt['run_root'])
    root.mkdir()
    launch = partial(start, module=rt['runner'])
    try:
        for action, names in (*initial_stages, ('smoke', recipes), ('fit', recipes), ('timing', ('source',)),
                              ('evaluate', ROUTES)):
            schedule(rt, [(action, name) for name in names], launch)
        if launch(rt, 'score', 'source').wait():
            raise RuntimeError('Independent LoRa source scoring failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root / 'state.json', dict(status='FAILED', error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pub = commands.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + key, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'fit', 'timing', 'evaluate', 'score'):
        cmd = commands.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        if action != 'dispatch':
            cmd.add_argument('name', choices=RECIPES if action in ('smoke', 'fit') else
                             ROUTES if action == 'evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args, run=RUN, runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action == 'dispatch':
        dispatch(rt)
    elif action == 'score':
        score_frozen(rt)
    else:
        activate(rt)
        if action == 'evaluate':
            evaluate(rt, args['name'])
        elif action == 'timing':
            measure(rt)
        else:
            fit(rt, smoke=action == 'smoke', recipe=RECIPES[args['name']], receiver_batches=True,
                calculate=partial(natural_objective, preserve=.35, make_pair=paired_echo_inputs))


if __name__ == '__main__':
    main()
