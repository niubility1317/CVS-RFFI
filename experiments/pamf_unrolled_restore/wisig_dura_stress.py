"""Run the matched expanded-DURA WiSig comparison and freeze results before scoring."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.evaluation import handle
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_repl_view_shuffle.runner import activate
from . import wisig_dura_stress_data as data
from . import wisig_dura_stress_fit as training
from .wisig_lora_source_fit import source_iq
from .wisig_preamble_dura_target import (
    BASELINE, COUNTS, F_RUN, ROWS, SEEDS, TASKS, VIEWS, readers, single_packet_median, view_root,
)
from .wisig_receiver_estimates import receive_coordinates
from .wisig_short_training_source import original_r69


RUN = '20261125-phase1-wisig-blind-capacity-stress-m2-r03'
SOURCE_RUN = '20261125-phase1-wisig-blind-capacity-stress-m2-r02'
METHODS = ('raw', 'R69', 'trained_default', 'trained_strong')
BATCH = 96
CASES = {f'{profile}-{key}': (profile, route, split) for profile in data.PROFILES
         for key, (route, split) in ROWS.items()}


def classify(heads, iq):
    values = []
    for head in heads.values():
        logits = head.classify_features(head.features(iq))
        if logits.shape != (len(iq), 6) or not torch.isfinite(logits).all():
            raise ValueError('Invalid scores from the frozen Main split reader')
        values.append(logits.argmax(1).cpu().tolist())
    return values


@torch.no_grad()
def smoke(rt, *, arms=data.PROFILES, load=training.frozen_fit, force_practical=False):
    activate(rt)
    root = Path(rt['run_root'])
    clean, selected, calibration = source_iq(Path(rt['source_root']))
    folder, quantile = data.read_plan(rt['source_root'], data.EPOCHS[0])
    indices = selected[quantile[selected] >= .575][:48]
    quiet = tensor(clean[calibration[:12]], 'cuda:0')
    observed = tensor(np.load(folder / 'full.npy', mmap_mode='r', allow_pickle=False)[indices], 'cuda:0')
    _, heads = readers(rt)
    r69 = original_r69(Path(rt['parent_root']))
    classify(heads, r69(observed[:12])[0])
    latency = {}
    for profile in arms:
        model, threshold = load(Path(rt.get('frozen_root', root)), profile)
        quiet_variance = receive_coordinates(quiet)[-1]
        unmodified, _ = training.restore(model, threshold, quiet, quiet_variance)
        if not torch.equal(unmodified, quiet):
            raise ValueError('Clean calibration samples must use the exact bypass')
        forced = receive_coordinates(observed)[-1]
        restored, active = training.restore(model, 0., observed, forced, force=force_practical)
        if force_practical and not bool(active.all()):
            raise ValueError('Forced satellite inference skipped an observation')
        if not torch.isfinite(restored).all():
            raise ValueError('Nonfinite frozen restorer output')
        classify(heads, restored)
        latency[profile] = dict(clean_bypass_ms=single_packet_median(
            lambda: training.restore(model, threshold, quiet[:1], receive_coordinates(quiet[:1])[-1])),
            active_restoration_ms=single_packet_median(
                lambda: training.restore(model, 0., observed[:1], receive_coordinates(observed[:1])[-1])),
            naturally_active=int((forced > threshold).sum()))
    write(root / 'checkpoint_smoke.json', dict(status='PASS', device=torch.cuda.get_device_name(),
          source_examples=len(quiet) + len(observed), target_reads=0, seeds=list(SEEDS), latency=latency,
          force_practical=force_practical))


@torch.no_grad()
def predict(rt, case, *, methods=METHODS, load=training.frozen_fit, force_practical=False,
            restore=training.restore, clean_passthrough=True, measure_iq=False):
    activate(rt)
    profile, route, split = CASES[case]
    root = Path(rt['run_root'])
    originals, heads = readers(rt)
    original = originals[SEEDS[0]]
    target = view_root(rt, original, route, split)
    if any(view_root(rt, entry, route, split) != target or entry['classes'] != original['classes']
           for entry in originals.values()):
        raise ValueError('Frozen readers disagree on the target physical data')
    manifest, ids, positions = handle(target, original['classes'])
    if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
        raise ValueError('Unexpected Main split IQ population')
    replay = rt.get('decision_replay')
    reference = Path(replay) / case if replay else Path(rt['baseline_replay']) / route / split
    closure = read(reference / 'complete.json')
    expected = dict(status='PREDICTIONS_FROZEN', route=route, split=split, count=len(ids),
                    seeds=list(SEEDS), views=list(VIEWS), truth_read=False)
    if replay and closure.get('profile')!=profile:
        raise ValueError('Historical baseline strength differs')
    if any(closure.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected the fixed historical raw/R69 comparison')
    with np.load(reference / 'decisions.npz', allow_pickle=False) as archive:
        if not np.array_equal(archive['ids'], ids):
            raise ValueError('Historical raw/R69 packet order differs')
        baseline = {key: archive[key] for key in archive.files if key.startswith(('raw__', 'R69__'))}
    strong = Path(rt.get('strong_view_root', root / 'strong_target_iq'))
    if profile == 'strong':
        if read(strong / 'complete.json') != dict(status='IQ_COMPLETE', count=168000, truth_read=False):
            raise ValueError('Harder IQ views are incomplete')
        _, strong_ids, strong_positions = handle(view_root(rt, original, 'full', split), original['classes'])
        if not np.array_equal(strong_ids, ids) or strong_positions is None:
            raise ValueError('Stress views must use the original global physical indices')
        if not replay:
            r69 = original_r69(Path(rt['parent_root']))
    frozen = Path(rt.get('frozen_root', root))
    models = {name:load(frozen, name.removeprefix('trained_')) for name in methods if name not in ('raw','R69')}
    if measure_iq:
        if not force_practical or clean_passthrough:
            raise ValueError('Paired IQ metrics require restoration on all inputs')
        clean_array = np.load(manifest['clean_ref'], mmap_mode='r', allow_pickle=False)
        r69 = original_r69(Path(rt['parent_root']))
    output, activation = {}, {}
    iq_metrics = {}
    started = perf_counter()
    for view in VIEWS:
        use_strong = profile == 'strong' and view != 'clean'
        view_positions = strong_positions if use_strong else positions
        if use_strong:
            file = strong / route / f'{view}.npy'
        elif view == 'clean':
            file = Path(manifest['clean_ref'])
        else:
            file = Path(manifest['observations_root']) / f'{view}.npy'
        observed = np.load(file, mmap_mode='r', allow_pickle=False)
        if observed.shape[1:] != (2, 256) or observed.dtype != np.float32:
            raise ValueError('Unexpected received IQ layout')
        if view_positions is not None and (view_positions.min() < 0 or view_positions.max() >= len(observed)):
            raise ValueError('Received IQ indices are out of range')
        if view_positions is None and len(observed) != len(ids):
            raise ValueError('Received IQ count differs from the split')
        decisions = {name: np.empty((len(SEEDS), len(ids)), dtype=np.uint8) for name in methods}
        predict_baselines = use_strong and not replay
        if not predict_baselines:
            for name in ('raw', 'R69'):
                decisions[name] = baseline[f'{name}__{view}']
        active_count = {name: 0 for name in models}
        sums = dict.fromkeys(methods, 0.)
        if force_practical and clean_passthrough and view=='clean':
            for name in models:
                decisions[name] = decisions['raw'].copy()
            output.update({f'{name}__{view}': value for name, value in decisions.items()})
            activation[view] = active_count
            print('WISIG_STRESS_PREDICTED', profile, route, split, view, len(ids), active_count, flush=True)
            continue
        for start in range(0, len(ids), BATCH):
            selection = slice(start, min(start + BATCH, len(ids)))
            rows = observed[selection if view_positions is None else view_positions[selection]]
            iq = tensor(rows, 'cuda:0')
            variance = None if force_practical else receive_coordinates(iq)[-1]
            if predict_baselines or measure_iq:
                original_restored = r69(iq)[0]
            if predict_baselines:
                decisions['raw'][:, selection] = classify(heads, iq)
                decisions['R69'][:, selection] = classify(heads, original_restored)
            if measure_iq:
                clean = tensor(clean_array[selection if positions is None else positions[selection]], 'cuda:0')
                energy = clean.square().mean((1, 2)).clamp_min(1e-4)
                for name, value in (('raw', iq), ('R69', original_restored)):
                    sums[name] += float(((value - clean).square().mean((1, 2)) / energy).sum())
            for name, (model, gate) in models.items():
                recovered, active = restore(model, gate, iq, variance, force=force_practical)
                if force_practical:
                    active_count[name] += int(active.sum().item())
                    decisions[name][:, selection] = classify(heads,recovered)
                    if measure_iq:
                        sums[name] += float(((recovered - clean).square().mean((1, 2)) / energy).sum())
                    continue
                mask = np.asarray(active.cpu().tolist(), dtype=bool)
                active_count[name] += int(mask.sum())
                values = decisions['raw'][:, selection].copy()
                if mask.any():
                    values[:, mask] = classify(heads, recovered[active])
                decisions[name][:, selection] = values
        output.update({f'{name}__{view}': value for name, value in decisions.items()})
        if force_practical and any(count!=len(ids) for count in active_count.values()):
            raise ValueError('Forced satellite inference skipped physical packets')
        activation[view] = active_count
        if measure_iq:
            iq_metrics[view] = {name: total / len(ids) for name, total in sums.items()}
        print('WISIG_STRESS_PREDICTED', profile, route, split, view, len(ids), active_count, flush=True)
    folder = root / 'predictions' / case
    folder.mkdir(parents=True)
    with (folder / 'decisions.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **output)
    marker = dict(expected, profile=profile, active=activation, force_practical=force_practical,
                  clean_passthrough=clean_passthrough, seconds=perf_counter()-started)
    if measure_iq:
        marker['mean_iq_nmse'] = iq_metrics
    write(folder / 'complete.json', marker)


def launch(rt, action, name, gpu=None):
    root = Path(rt['run_root'])
    runtime_path = (root / 'source_runtime.json' if action in ('fit-smoke', 'fit', 'source-iq')
                    else Path(rt['release_root']) / 'runtime.json')
    command = [rt['python'], '-u', '-m', rt['runner'], action, str(runtime_path), name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (root / 'logs' / f'{action}-{name}.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
    print('WISIG_STRESS_STARTED', action, name, process.pid, gpu, flush=True)
    return process


def cpu_stage(rt, stage):
    if launch(rt, stage, 'cpu').wait():
        raise RuntimeError(f'{stage} failed. Run artifacts are preserved')


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    state = root / 'state.json'
    command = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits']
    rows = [tuple(map(int, row.split(','))) for row in subprocess.check_output(command, text=True).splitlines()]
    rt = dict(rt, allowed_gpus=[i for _, _, i in sorted((load, -free, index) for index, load, free in rows)])
    try:
        if 'frozen_root' not in rt:
            write(root / 'source_runtime.json', {key: rt[key] for key in ('run_root', 'source_root')})
            write(state, dict(status='SOURCE_SMOKE'))
            schedule(rt, [('fit-smoke', 'default')], launch)
            write(state, dict(status='SOURCE_STRONG_IQ'), exclusive=False)
            cpu_stage(rt, 'source-iq')
            schedule(rt, [('fit-smoke', 'strong')], launch)
            write(state, dict(status='SOURCE_TRAINING'), exclusive=False)
            schedule(rt, [('fit', profile) for profile in data.PROFILES], launch)
        write(state, dict(status='FROZEN_CHECKPOINT_SMOKE'), exclusive=False)
        schedule(rt, [('smoke', 'checkpoints')], launch)
        write(state, dict(status='BLIND_TARGET_IQ'), exclusive=False)
        prepare = launch(rt, 'target-iq', 'cpu')
        schedule(rt, [('predict', case) for case, (profile, _, _) in CASES.items()
                      if profile == 'default'], launch)
        if prepare.wait():
            raise RuntimeError('target-iq failed. Run artifacts are preserved')
        write(state, dict(status='TARGET_PREDICTING'), exclusive=False)
        schedule(rt, [('predict', case) for case, (profile, _, _) in CASES.items()
                      if profile == 'strong'], launch)
        write(state, dict(status='PREDICTIONS_FROZEN', truth_read=False), exclusive=False)
        cpu_stage(rt, 'score')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    rt = {key: ctx[key] for key in ('project', 'python', 'frozen_sources')}
    base = PurePosixPath(rt['project'])
    runs = base / 'runs'
    channel_release = base / 'releases' / 'cvs_residual_sixscene_20261002_r01'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(runs / RUN), release_root=str(base / 'releases' / f'{RUN}-release'),
              frozen_root=str(runs / SOURCE_RUN),
              source_root=str(runs / '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02' / 'cache'),
              parent_root=str(runs / '20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'),
              baseline_root=str(runs / BASELINE),
              baseline_replay=str(runs / '20261124-phase1-wisig-preamble-dura-main-m4-r01'),
              residual_reference=str(runs / '20261002-phase1-cvs-residual-sixscene-manysig-m8-r01' / 'received_views'),
              channel_release=str(channel_release), recognizer_run=str(runs / F_RUN),
              external_releases=[str(channel_release), str(runs / SOURCE_RUN)], max_workers=4,
              max_compute_processes=None)
    from experiments.pamf_lora_main import publish as transport

    extras = ('tools/wisig_paired_main_uncertainty.py', 'tools/r17_activation_factorization.py')
    transport.RELEASE_CODE = transport.RELEASE_CODE.replace(
        "'experiments/pamf_lora_main', entry",
        "'experiments/pamf_lora_main', 'experiments/pamf_unrolled_restore', *rt['package_tools']")
    transport.RELEASE_CODE = transport.RELEASE_CODE.replace(
        "with (release/'controller.log').open('x') as log:",
        "subprocess.run([*cmd[:4], '--help'], cwd=release, env=env, check=True, stdout=subprocess.DEVNULL)\n"
        "with (release/'controller.log').open('x') as log:")
    release(repo, output, dict(rt, package_tools=list(extras)), ssh_config, archive_extras=extras)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        command.add_argument('--' + key, type=Path, required=True)
    jobs = dict(dispatch=dispatch, smoke=lambda rt, _: smoke(rt), predict=predict,
                **{'source-iq': lambda rt, _: data.source_cache(rt),
                   'target-iq': lambda rt, _: data.target_cache(rt),
                   'fit-smoke': lambda rt, name: training.fit(rt, name, smoke=True),
                   'fit': training.fit})
    for key in (*jobs, 'score'):
        command = sub.add_parser(key)
        command.add_argument('runtime', type=Path)
        if key != 'dispatch':
            command.add_argument('name')
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'publish':
        publish(**args)
    else:
        rt = read(args['runtime'])
        if action == 'score':
            from .wisig_dura_stress_score import score

            score(rt)
        elif action == 'dispatch':
            dispatch(rt)
        else:
            jobs[action](rt, args['name'])


if __name__ == '__main__':
    main()
