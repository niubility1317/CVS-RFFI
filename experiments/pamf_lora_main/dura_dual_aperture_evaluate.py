"""Replay one frozen dual-aperture checkpoint on the two registered LoRa views."""

import argparse
from functools import partial
from json import JSONDecodeError
from pathlib import Path, PurePosixPath
import sys
from time import sleep


if __name__ == '__main__' and __spec__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import dura_capacity_target as expanded, weak_guard_target as replay
from experiments.pamf_lora_main.dura_dual_aperture_train import RUN as ORIGIN


RUN = '20261125-phase1-lora-dura-dual-aperture-pilot-s2026092702-r01'
ORIGIN_CODE = '760d7466b64ae89b6806972bdc52792ae6b49f1a'
SEED = 2026092702
SEEDS = (SEED,)
PER_RECEIVER = 200
MODE = 'dual_aperture_half'
MODES = ('direct', MODE)
SOURCE_MODES = ('direct', 'dura_half', 'expanded_half', MODE)
SOURCE_COMPARISONS = tuple((MODE, control) for control in SOURCE_MODES[:-1])
ROUTES = replay.ROUTES
EVENTS = 'LORA_DUAL_APERTURE_PILOT'


def completion():
    return replay.completion(origin=ORIGIN, origin_code=ORIGIN_CODE, modes=MODES,
                             seeds=SEEDS, per_receiver=PER_RECEIVER)


def expected_files():
    return replay.expected_files(MODES, seeds=SEEDS)


def source_root(rt):
    project = Path(rt['project'])
    archived = read(project / 'releases' / f'{ORIGIN}-release' / 'runtime.json')
    root = project / 'runs' / ORIGIN
    if (archived['run_id'], archived['code_commit'], Path(archived['run_root'])) != (ORIGIN, ORIGIN_CODE, root):
        raise ValueError('Expected the preregistered dual-aperture training identity')
    if Path(archived['source_root']) != Path(rt['source_root']):
        raise ValueError('The source IQ contract binding changed')
    return root


def check_frozen(root, recipe_name, steps):
    if read(root/'state.json') != dict(status='SOURCE_FROZEN', steps=steps):
        raise ValueError('The source optimizer must reach its frozen endpoint before evaluation')
    terminal = read(root/f'{recipe_name}-complete.json')
    if (terminal['status'], terminal['estimator'], terminal['steps']) != ('SOURCE_FROZEN', recipe_name, steps):
        raise ValueError('Unexpected source terminal checkpoint')
    return terminal


def load_restoration(rt, device):
    from experiments.pamf_lora_main.dura_apertures import ApertureTrackers, RECIPE
    from experiments.pamf_lora_main.full_packet_views import source_contract
    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_lora_main.spectral_block_source import load_models

    root = source_root(rt)
    terminal = check_frozen(root, RECIPE.name, RECIPE.steps)
    activate(rt)
    contract = source_contract(Path(rt['source_root']))
    (model,), threshold = load_models(root, contract, device, {RECIPE.name: RECIPE})
    if threshold != terminal['threshold']:
        raise ValueError('The pinned source threshold and checkpoint disagree')
    trackers = ApertureTrackers().to(device).eval()

    def restore(iq):
        restore, active = model.restore(trackers, iq, threshold)
        return restore, int(active.sum())

    return restore, contract


def bind(rt, device):
    from experiments.pamf_lora_main.complex_vector_source import load_reader

    restore, contract = load_restoration(rt, device)
    return {SEED: load_reader(Path(rt['run_root']), contract, device)}, restore, contract


def bind_source_reference(rt, route, contract, device):
    from experiments.pamf_lora_main.dura_repair import PacketRestorer
    from experiments.pamf_lora_main.full_dura_detail import half_baseline, phase_baseline_models
    from experiments.pamf_lora_main.full_supervised import handles
    from experiments.pamf_lora_main import residual_dura as residual

    root = Path(rt['run_root'])
    source = Path(rt['source_root'])
    if route == 'full':
        _, cache, reference_contract = handles(dict(rt, parent_root=str(root.parent/replay.DATA)))
        if contract != reference_contract:
            raise ValueError('The registered Full source cache and checkpoint differ')
        base, prior = phase_baseline_models(root, contract, SEED, device)
        reference = partial(half_baseline, base, prior)
    else:
        cache = None
        g, _, projector = residual.frozen_models(rt, root.parent/residual.RUN, (), device)

        def reference(iq):
            flat = iq.flatten(0, 1)
            projected = projector.restore(flat, g(flat)).reshape_as(iq)
            return iq + .5*(projected-iq), None

    return source, cache, contract, reference, PacketRestorer('consensus').to(device).eval()


def load_expanded(rt, contract, device):
    from experiments.pamf_lora_main.dura_capacity_source import RECIPES
    from experiments.pamf_lora_main.spectral_block_source import load_models

    root = Path(rt['run_root'])
    project = root.parent.parent
    archived = read(project/'releases'/f'{expanded.ORIGIN}-release'/'runtime.json')
    if (archived['run_id'], archived['code_commit']) != (expanded.ORIGIN, expanded.ORIGIN_CODE):
        raise ValueError('Unexpected expanded DURA checkpoint ancestry')
    reference_root = root.parent/expanded.ORIGIN
    summary = expanded.frozen_fit(reference_root, RECIPES['expanded'].steps)
    models, threshold = load_models(reference_root, contract, device, {'expanded': RECIPES['expanded']})
    if summary['threshold'] != threshold:
        raise ValueError('The expanded DURA source threshold differs')
    return models, threshold


def bind_source_controls(rt, route, contract, device):
    from experiments.pamf_lora_main.dura_capacity_source import READOUTS, recover

    binding = bind_source_reference(rt, route, contract, device)
    tracker = binding[-1]
    models, threshold = load_expanded(rt, contract, device)

    def expanded_half(iq):
        waves, _, _ = recover(iq, tracker, models, threshold)
        return waves[READOUTS.index('half')]

    return binding, expanded_half


def predict_source(rt, route):
    import torch
    from experiments.pamf_lora_main.channel_blind_source import panel, receive_only
    from experiments.pamf_lora_main.complex_vector_source import score
    from experiments.pamf_lora_main.local_wiener_source import freeze

    device = 'cuda:0'
    candidate, contract = load_restoration(rt, device)
    binding, expanded_half = bind_source_controls(rt, route, contract, device)

    def render(iq, _unused, own_reference, _tracker, _models):
        baseline, _ = own_reference(iq)
        old = expanded_half(iq)
        out, active = candidate(iq)
        return torch.stack((iq, baseline, old, out)), active

    score(rt, route, panel=panel, modes=SOURCE_MODES, render=render,
          binding=binding, frozen_models=(),
          require_clean_bypass=False, replay_fn=receive_only, finalize=freeze)


def score_source(rt):
    from experiments.pamf_lora_main.local_wiener_source import score_frozen

    score_frozen(rt, SOURCE_MODES, comparisons=SOURCE_COMPARISONS)
    rows = {}
    control, candidate = (SOURCE_MODES.index(k) for k in ('dura_half', MODE))
    root = Path(rt['run_root'])
    for route in ROUTES:
        result = read(root/f'{route}-source-scores.json')
        six, clean = result['six'], result['metrics']['clean']['correct']
        rows[route] = dict(six_net=six[candidate]-six[control], clean_net=clean[candidate]-clean[control])
    write(root/'source-promotion.json', dict(passed=all(x['six_net'] > 0 and x['clean_net'] >= 0
                                                       for x in rows.values()), pairs=rows, target_reads=0))


def wait_for_freeze(rt, _root, state):
    parent = source_root(rt) / 'state.json'
    write(state, dict(status='WAITING_FOR_FROZEN_TRAINING'))
    while True:
        try:
            status = read(parent)['status']
        except (FileNotFoundError, JSONDecodeError):
            status = None
        if status == 'SOURCE_FROZEN':
            return
        if status == 'FAILED':
            raise RuntimeError('The dual-aperture source run failed before freezing')
        cmdline = Path(f"/proc/{rt['training_dispatcher']}/cmdline")
        try:
            identity = cmdline.read_bytes()
        except FileNotFoundError:
            identity = b''
        if f'{ORIGIN}-release'.encode() not in identity:
            if read(parent)['status'] == 'SOURCE_FROZEN':
                return
            raise RuntimeError('The training dispatcher exited without a frozen model')
        sleep(20)


def source_panel(launch, state):
    from experiments.pamf_lora_main.refine_runner import select_gpu

    first = select_gpu()
    second = select_gpu(exclude=[first])
    gpus = dict(zip(ROUTES, (first, second), strict=True))
    write(state, dict(status='SOURCE_V_PREDICT', gpus=gpus), exclusive=False)
    jobs = [launch('source-predict', route, gpu) for route, gpu in gpus.items()]
    if any([job.wait() for job in jobs]):
        raise RuntimeError('Source V predictions failed')
    write(state, dict(status='SOURCE_V_SCORE'), exclusive=False)
    if launch('source-score', 'both', None).wait():
        raise RuntimeError('Independent source V score failed')


def publish(repo, output, context, ssh_config, training_receipt):
    from experiments.pamf_lora_main.publish import release, release_commit

    cfg = read(context)
    receipt = read(training_receipt)
    project = PurePosixPath(cfg['project'])
    if (PurePosixPath(receipt['run_root']) != project/'runs'/ORIGIN or receipt['code_commit'] != ORIGIN_CODE
            or receipt['archive_transfer'] != 'VERIFIED'):
        raise ValueError('Expected the verified dual-aperture training receipt')
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo), training_dispatcher=receipt['pid'],
              release_root=str(project/'releases'/f'{RUN}-release'), run_root=str(project/'runs'/RUN),
              source_root=str(project/'releases'/f'{replay.DATA}-release'/'data'/'source'),
              external_releases=[str(project/'releases'/f'{ORIGIN}-release')])
    release(repo, output, rt, ssh_config)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for flag in ('repo', 'output', 'context', 'ssh-config', 'training-receipt'):
        pub.add_argument('--' + flag, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'source-predict', 'source-score'):
        parser = actions.add_parser(action)
        parser.add_argument('runtime', type=Path)
        if action != 'dispatch':
            parser.add_argument('shard')
    args = vars(cli.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
        return
    rt = read(args['runtime'])
    if rt['run_id'] != RUN:
        raise ValueError('Expected the immutable dual-aperture evaluation root')
    if action == 'dispatch':
        replay.dispatch(rt, args['runtime'], entrypoint=Path(__file__).resolve(), modes=MODES, complete=completion,
                        events=EVENTS, seeds=SEEDS, before_smoke=partial(wait_for_freeze, rt), after_smoke=source_panel)
    elif action == 'source-predict':
        predict_source(rt, args['shard'])
    elif action == 'source-score':
        score_source(rt)
    else:
        replay.run_worker(rt, action, args['shard'], bind=bind, modes=MODES, events=EVENTS, seeds=SEEDS,
                          per_receiver=PER_RECEIVER, time_restoration=True)


if __name__ == '__main__':
    main()
