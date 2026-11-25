"""Isolate first-stage design choices in a frozen WiSig native FIR model."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import monotonic, sleep

from experiments.cric_dura_comparison.io import read, tensor, write


RUN = '20261125-diagnostic-wisig-native-fir-anchor-source-m2-r01'
PRECISION_RUN = '20261125-diagnostic-wisig-native-fir-first-map-precision-source-m2-r01'
PARENT = '20261125-phase1-wisig-native-fir-source-m4-r01'
PARENT_COMMIT = 'ef9f4c227ce57c79dc17a3aeb599395cc25d883b'
ROUTES = ('full', 'residual')
ARMS = ('learned_anchor', 'received_anchor')
PRECISION_ARMS = ('original_precision', 'first_map_8x_precision')


def modes_for(arms):
    return ('direct', 'r69', *(f'{arm}_{stage}' for arm in arms for stage in ('map_1', 'final')))


def reference(rt):
    parent = read(Path(rt['reference']) / 'runtime.json')
    if (parent['run_id'], parent['code_commit'], parent['project']) != (PARENT, PARENT_COMMIT, rt['project']):
        raise ValueError('Unexpected frozen native FIR source release')
    if read(Path(parent['run_root']) / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Completed native FIR source run required')
    return parent


def models_from(rt):
    from experiments.pamf_unrolled_restore.wisig_native_fir_source import build, runner

    (model, _), = runner.load_models(reference(rt), arms=('centered',), build=build)
    return ((model, False), (model, True))


def precision_readouts(model, analysis, first_multiplier):
    import torch
    from experiments.pamf_r65_inference.program import reused_map
    from experiments.pamf_unrolled_restore.wisig_native_fir import SCALE, SAMPLES, WindowAction
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex, as_iq

    value, level, base, kernel, main, image, variance = analysis
    action = WindowAction(kernel, main, image, model.support)
    iq = as_iq(value)
    current = as_complex(base)
    logvar = variance.log().clamp(-16, 4)[:, :, None].expand(-1, -1, SAMPLES) / 8
    stages = []
    for network, multiplier in zip(model.priors, (first_multiplier, 1), strict=True):
        innovation = action.adjoint(value - action(current))
        features = as_iq(innovation)
        features = features / (features.square().mean((1, 2), keepdim=True) + 1e-6).sqrt()
        prior = as_complex(base + SCALE * network(torch.cat((iq, base, as_iq(current), features, logvar), 1)))
        current = reused_map(action, prior, current, variance / multiplier, innovation)
        stages.append(current * level)
    return tuple(stages)


def proposals(model, alternative, received, *, precision_probe=False):
    import torch
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_iq

    analysis = model.analyze(received)
    if precision_probe and alternative:
        readouts = precision_readouts(model, analysis, 8)
    else:
        value, level, base, *estimates = analysis
        if alternative:
            base = as_iq(value)
        stages, _ = model.stages((value, level, base, *estimates))
        readouts = (stages[2], stages[4])
    count = len(received)
    active = torch.ones(count, dtype=torch.bool, device=received.device)
    offsets = torch.zeros(count, dtype=torch.int64, device=received.device)
    return readouts, active, offsets


def smoke(rt, original, *, precision_probe):
    import torch
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import classify, core, tri
    from experiments.pamf_unrolled_restore.wisig_native_fir import disturb
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex

    with torch.no_grad():
        source, _, indices = core.source_iq(Path(original['source_root']))
        received = as_complex(tensor(source[indices[:8]], 'cuda:0'))
        reader = tri.reader_factory(original).eval().requires_grad_(False)
        (model, _), _ = models_from(rt)
        perturbed = disturb(received, torch.Generator(device='cuda:0').manual_seed(2026112507), model)
        for y in (received, perturbed):
            analysis = model.analyze(y)
            expected, *_ = model.restore(analysis)
            if precision_probe:
                for stage, old in zip(precision_readouts(model, analysis, 1), (expected[2], expected[4]), strict=True):
                    torch.testing.assert_close(stage, old, rtol=5e-5, atol=5e-6)
            for alternative in (False, True):
                stages, active, _ = proposals(model, alternative, y, precision_probe=precision_probe)
                if not bool(active.all()) or any(x.shape != y.shape for x in stages):
                    raise ValueError('Every packet must complete both MAP stages')
                for stage in stages:
                    classify(reader, stage)
                if not alternative:
                    torch.testing.assert_close(stages[1], expected[4], rtol=5e-5, atol=5e-6)
                elif not bool((stages[1] - y).abs().amax(-1).gt(0).all()):
                    raise ValueError('Frozen MAP stages must update each smoke waveform')
                elif precision_probe and y is perturbed and torch.allclose(stages[0], expected[2]):
                    raise ValueError('The first-stage observation weight must change the disturbed readout')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', count_per_view=8, source_V_reads=0,
           target_reads=0, all_packets_recovered=True))


def freeze_predictions(root, parent_root, views, original_modes, *, arms=ARMS):
    import numpy as np

    old_indices = [original_modes.index(name) for name in ('direct', 'r69', 'centered_map_1', 'centered_final')]
    modes = modes_for(arms)
    summaries = {}
    for route in ROUTES:
        by_route = read(root / f'{route}-complete.json')
        if set(by_route) != {f'{route}_{view}' for view in views}:
            raise ValueError('Incomplete source route')
        for key, details in by_route.items():
            with np.load(root / f'{key}.npz', allow_pickle=False) as row, np.load(
                    parent_root / f'{key}.npz', allow_pickle=False) as frozen:
                if (not np.array_equal(row['ids'], frozen['ids']) or row['decision'].shape != (len(modes), 1800)
                        or not np.array_equal(row['decision'][:4], frozen['decision'][old_indices])):
                    raise ValueError(f'Archived comparison failed to reproduce for {key}')
            if details['count'] != 1800 or details['active'] != dict.fromkeys(arms, 1800):
                raise ValueError(f'Missing all-packet restoration for {key}')
        summaries.update(by_route)
    write(root / 'complete.json', dict(status='FROZEN', count=1800, modes=list(modes), views=list(views),
          source_truth_reads=0, target_reads=0, summaries=summaries))


def worker(rt, action):
    from functools import partial

    from experiments.pamf_lora_main.runner import activate
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import core, evaluate
    from experiments.pamf_unrolled_restore.wisig_native_fir_source import MODES as ORIGINAL_MODES

    original = reference(rt)
    current = dict(original, run_root=rt['run_root'])
    precision_probe = rt['run_id'] == PRECISION_RUN
    arms = PRECISION_ARMS if precision_probe else ARMS
    if action == 'score':
        control, alternative = arms
        comparisons = ((f'{alternative}_map_1', f'{control}_map_1'),
                       (f'{alternative}_final', f'{control}_final'),
                       (f'{alternative}_final', 'direct'), (f'{alternative}_final', 'r69'))
        core.baseline.score(current, modes_for(arms), comparisons=comparisons)
        return
    activate(original)
    smoke(rt, original, precision_probe=precision_probe)
    root = Path(rt['run_root']) / 'predictions'
    root.mkdir()
    for route in ROUTES:
        evaluate(current, route, arms=arms, models_from=lambda _: models_from(rt),
                 variants=partial(proposals, precision_probe=precision_probe))
    freeze_predictions(root, Path(original['run_root']) / 'predictions', core.baseline.VIEWS, ORIGINAL_MODES, arms=arms)


def dispatch(runtime):
    from experiments.pamf_lora_main.runner import available

    rt = read(runtime)
    reference(rt)
    root = Path(rt['run_root'])
    root.mkdir()

    def status(value, **fields):
        write(root / 'state.json', dict(status=value, **fields), exclusive=False)

    def execute(action, gpu=None):
        reference_root = rt['reference']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=reference_root, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        cmd = [rt['python'], '-u', str(Path(__file__).resolve()), action, str(runtime)]
        with (root / f'{action}.log').open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(cmd, env=env, cwd=reference_root, stdout=stream, stderr=subprocess.STDOUT)
            print('WISIG_ANCHOR_WORKER', action, child.pid, gpu, flush=True)
            if child.wait():
                raise ValueError(f'WiSig anchor {action} worker failed')

    try:
        status('WAIT_GPU')
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()
        devices = [tuple(map(int, row.split(','))) for row in rows]
        ranked = [i for _, _, i in sorted((util, -free, i) for i, util, free in devices)]
        deadline = monotonic() + 1800
        while not (cards := available(ranked, rt['max_compute_processes'])):
            if monotonic() >= deadline:
                raise ValueError('Available GPU wait expired')
            sleep(20)
        gpu = next(i for i in ranked if i in cards)
        status('SOURCE_PREDICTING', gpu=gpu)
        execute('predict', gpu)
        status('PREDICTIONS_FROZEN')
        execute('score')
        status('SOURCE_SCORED')
    except Exception as exc:
        status('FAILED', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch', 'predict', 'score'):
        actions.add_parser(action).add_argument('runtime', type=Path)
    publish = actions.add_parser('publish')
    publish.add_argument('--first-map-precision', action='store_true')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        publish.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        from experiments.pamf_lora_main.publish import release, release_commit

        context = read(args.context)
        rt = {key: context[key] for key in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        parent = str(project / 'releases' / f'{PARENT}-release')
        name = PRECISION_RUN if args.first_map_precision else RUN
        rt.update(run_id=name, runner=__spec__.name, code_commit=release_commit(args.repo), reference=parent,
                  release_root=str(project / 'releases' / f'{name}-release'), run_root=str(project / 'runs' / name),
                  external_releases=[parent, str(project / 'runs' / PARENT)], max_compute_processes=8)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    else:
        worker(read(args.runtime), args.action)


if __name__ == '__main__':
    main()
