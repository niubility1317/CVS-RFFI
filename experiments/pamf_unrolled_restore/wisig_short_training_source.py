"""Freeze WiSig engineering controls, predict on source V, then score separately."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import perf_counter, sleep

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_fcrossrx_readout.source_diagnostic import VIEWS, source_reference
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.refine_runner import start
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_r65_four_route.runtime import load_checkpoint
from experiments.pamf_stage2_inference.engine import InferenceEngine

from .design import KIND, RUN as R17_RUN
from .wisig_short_training import READOUTS, SAMPLE_RATE_HZ, readouts


RUN = '20261123-phase1-wisig-stf-engineering-source-s2026092701-r02'
CACHE_RUN = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
R69_RUN = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
R69_COMMIT = 'b2a2ce937a64654e17e84c2aa817b9da4257a48f'
ROUTES = {'full': 'validation_full', 'residual': 'validation'}
MODES = (*READOUTS, 'r69')
BATCH = 64


def handles(rt):
    cache = Path(rt['source_root'])
    contract = read(cache / 'complete.json')
    if (cache.parent.name != CACHE_RUN or contract['source_V_count'] != 1800
            or contract['TX_labels_read'] or contract['target_reads']
            or contract['paired_full_residual'] is not True):
        raise ValueError('Expected the existing TX-blind, dual-route WiSig source V panel')
    parent = Path(rt['parent_root'])
    if parent.name != R17_RUN:
        raise ValueError('Expected original WiSig source contract and frozen F50 binding')
    args = read(parent / 'launch.json')['argv']
    source, reader = [Path(args[args.index(key) + 1]) for key in ('--source-cache', '--reader-folder')]
    ids, metadata, positions = source_reference(cache / ROUTES['residual'], source)
    full_ids, _, _ = source_reference(cache / ROUTES['full'], source)
    if not np.array_equal(ids, full_ids):
        raise ValueError('Full and residual WiSig source V IDs differ')
    return cache, parent, source, reader, ids, metadata.rx[positions], positions


def quantiles(values):
    return np.quantile(np.concatenate(values), (0.1, 0.5, 0.9)).tolist()


def original_r69(parent):
    r69, proof = load_checkpoint(parent.parent / R69_RUN / 'stage2' / KIND / 'final.pt', 'cuda:0')
    if ((proof['arm'], proof['epoch'], proof['updates'], proof['code_commit']) != ('stage2', 20, 4940, R69_COMMIT)
            or sum(p.numel() for p in r69.parameters()) != 266951):
        raise ValueError('Expected original frozen R69 checkpoint')
    return InferenceEngine(r69.eval().requires_grad_(False), reuse=True, graph=True, native=False)


@torch.no_grad()
def predict(rt, *, smoke=False, transform=readouts, readout_names=READOUTS, reader_factory=None):
    cache, parent, source, reader, ids, _, _ = handles(rt)
    f = load_reader(reader, source, 'cuda:0') if reader_factory is None else reader_factory(rt)
    f.eval().requires_grad_(False)
    classify = f if reader_factory is None else lambda y: f.classify_features(f.features(y))
    engine = original_r69(parent)
    root = Path(rt['run_root'])
    modes = (*readout_names, 'r69')
    numbers = {}
    for route, folder in ROUTES.items():
        numbers[route] = {}
        for scene in VIEWS:
            path = cache / folder / f'{scene}.npy'
            observed = np.load(path, mmap_mode='r', allow_pickle=False)
            if observed.shape != (len(ids), 2, 256) or observed.dtype != np.float32:
                raise ValueError(f'Unexpected shape or dtype for {path}')
            predictions = []
            measures = {k: [] for k in ('coherence_lag20', 'coherence_lag60', 'effective_variance',
                        'variance_over_received_power', 'cfo_fine_minus_coarse_hz', 'apparent_clock_ppm', 'clock_fit_rmse_rad')}
            started = perf_counter()
            for i in range(0, BATCH if smoke else len(ids), BATCH):
                x = observed[i:i + BATCH]
                variants, est = transform(x)
                for key, values in zip(measures, (
                    est.coarse_coherence, est.long_coherence, est.effective_variance,
                    est.effective_variance / np.maximum(est.effective_variance + est.signal_power, 1e-12),
                    (est.fine_rad_per_sample - est.coarse_rad_per_sample) * SAMPLE_RATE_HZ / (2 * np.pi),
                    1e6 * est.clock_delay_per_sample, est.clock_phase_rmse,
                )):
                    measures[key].append(values)
                y = tensor(x, 'cuda:0')
                restored, = engine(y)
                inputs = (y, *(tensor(variants[m], 'cuda:0') for m in readout_names[1:]), restored)
                logits = torch.stack([classify(z) for z in inputs])
                if logits.shape != (len(modes), len(x), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Incomplete or nonfinite frozen source decisions')
                predictions.append(logits.argmax(dim=-1).tolist())
            numbers[route][scene] = dict(seconds=perf_counter() - started,
                                         **{k: quantiles(v) for k, v in measures.items()})
            if not smoke:
                path = root / 'predictions' / f'{route}_{scene}.npz'
                path.parent.mkdir(exist_ok=True)
                with path.open('xb') as stream:
                    np.savez(stream, ids=ids, decision=np.concatenate(predictions, axis=1).astype(np.uint8))
                print('WISIG_STF_FROZEN', route, scene, len(ids), flush=True)
    write(root / ('smoke.json' if smoke else 'predictions/complete.json'),
          dict(status='PASS' if smoke else 'FROZEN', summaries=numbers, count=BATCH if smoke else len(ids),
               modes=list(modes), views=list(VIEWS), source_truth_reads=0, target_reads=0))


def score(rt, modes=MODES):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU truth-side scorer required')
    _, _, source, _, ids, rx, positions = handles(rt)
    root = Path(rt['run_root']) / 'predictions'
    complete = read(root / 'complete.json')
    if any(complete[k] != v for k, v in dict(status='FROZEN', modes=list(modes), views=list(VIEWS),
                                            count=len(ids), source_truth_reads=0, target_reads=0).items()):
        raise ValueError('Complete fixed source predictions are required before opening labels')
    records = {}
    for route in ROUTES:
        decisions = []
        for scene in VIEWS:
            with np.load(root / f'{route}_{scene}.npz', allow_pickle=False) as rows:
                pred = rows['decision']
                if (not np.array_equal(rows['ids'], ids) or pred.shape != (len(modes), len(ids))
                        or pred.dtype != np.uint8 or (pred > 5).any()):
                    raise ValueError('Prediction decisions or physical IDs differ from frozen panel')
                decisions.append(pred)
        records[route] = np.stack(decisions)
    labels = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[positions]
    scores = {}
    groups = {'all': np.ones(len(rx), dtype=bool), **{str(x): rx == x for x in np.unique(rx)}}
    for route, decisions in records.items():
        scores[route] = {}
        for group, mask in groups.items():
            hit = decisions[:, :, mask] == labels[mask]
            scores[route][group] = dict(count=int(mask.sum()), correct=hit.sum(axis=-1).tolist(), pairs={})
            for base in ('direct', 'r69'):
                baseline = hit[:, modes.index(base):modes.index(base) + 1]
                scores[route][group]['pairs'][base] = dict(
                    repaired=(hit & ~baseline).sum(axis=-1).tolist(), damaged=(~hit & baseline).sum(axis=-1).tolist())
    write(root.parent / 'source-scores.json', dict(status='SOURCE_SCORED', scores=scores, views=list(VIEWS),
           modes=list(modes), reader_seed=2026092701, target_reads=0, truth_last=True))


def select_gpu(exclude=()):
    query = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits']
    for _ in range(30):
        rows = (tuple(map(int, line.split(','))) for line in subprocess.check_output(query, text=True).splitlines())
        available = [(load, -free, gpu) for gpu, load, free in rows if free >= 12_288 and gpu not in exclude]
        if available:
            return min(available)[-1]
        sleep(20)
    raise RuntimeError('No GPU has 12 GiB free')


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        for step in ('smoke', 'evaluate', 'score'):
            gpu = None if step == 'score' else select_gpu()
            write(root / 'state.json', dict(status=step.upper(), gpu=gpu), exclusive=False)
            if start(rt, step, 'stf', gpu, module=rt['runner']).wait():
                raise RuntimeError(f'WiSig source {step} failed')
        write(root / 'state.json', dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(root / 'state.json', dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    config = read(context)
    runtime = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(runtime['project'])
    runtime.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
                   release_root=str(project / 'releases' / f'{RUN}-release'), run_root=str(project / 'runs' / RUN),
                   source_root=str(project / 'runs' / CACHE_RUN / 'cache'), parent_root=str(project / 'runs' / R17_RUN))
    release(repo, output, runtime, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
    cmd = parser.parse_args()
    if cmd.action == 'publish':
        publish(cmd.repo, cmd.output, cmd.context, cmd.ssh_config)
        return
    runtime = read(cmd.runtime)
    if cmd.action == 'dispatch':
        dispatch(runtime)
    elif cmd.action == 'score':
        score(runtime)
    else:
        activate(runtime)
        predict(runtime, smoke=cmd.action == 'smoke')


if __name__ == '__main__':
    main()
