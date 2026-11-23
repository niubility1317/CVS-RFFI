"""Isolated source-V ablation using the paired clean IQ. Never import in a candidate."""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate

from . import wisig_short_training_source as core
from .wisig_timing import estimate_timing, realign
from .wisig_timing_source import RUN as TIMING


RUN = '20261123-diagnostic-wisig-source-paired-time-edge-s2026092701-r01'
MODES = ('direct', 'r69', 'received_timing', 'paired_timing', 'paired_timing_tail', 'paired_timing_gain')
SHIFTS = np.arange(-64, 65)


def paired_delay(obs, clean):
    size = obs.shape[1]
    score = np.empty((len(obs), len(SHIFTS)), dtype=np.float32)
    zero = np.arange(len(obs))
    for i, delta in enumerate(SHIFTS):
        x = clean[:, max(0, -delta):min(size, size - delta)]
        y = obs[:, max(0, delta):min(size, size + delta)]
        score[:, i] = np.abs((x.conj() * y).sum(axis=-1))**2 / np.maximum(
            (np.abs(x)**2).sum(axis=-1) * (np.abs(y)**2).sum(axis=-1), 1e-12)
    highest = score.argmax(axis=1)
    return SHIFTS[highest], score[zero, highest], score[:, len(SHIFTS) // 2]


def oracle_readouts(obs, clean):
    delta, after, before = paired_delay(obs, clean)
    det, margin, estimates = estimate_timing(obs)
    # The deployed hypothesis searches ±40. The diagnostic may give up to ±64.
    # A wider offset can leave too short an L-LTF prefix to extrapolate safely.
    output = realign(obs, delta)
    supported = np.abs(delta) <= 40
    output[supported] = realign(obs[supported], delta[supported], protocol=True,
                                cfo=estimates.coarse_rad_per_sample[supported])
    positions = np.arange(obs.shape[1]) + delta[:, None]
    valid = (positions >= 0) & (positions < obs.shape[1])
    # Only these two isolated diagnostics receive paired clean values after alignment.
    target_tail = np.where(valid, output, clean)
    gain = (clean * output.conj() * valid).sum(axis=1) / np.maximum(
        (np.abs(output)**2 * valid).sum(axis=1), 1e-12)
    target_gain = np.where(valid, output * gain[:, None], clean)
    return (output, target_tail, target_gain), dict(paired_delta=delta, received_delta=det,
                                                    timing_margin=margin, aligned_cos2=after, original_cos2=before)


@torch.no_grad()
def evaluate(rt, *, smoke=False):
    cache, _, source, reader, ids, _, _ = core.handles(rt)
    f = core.load_reader(reader, source, 'cuda:0').eval().requires_grad_(False)
    root = Path(rt['run_root'])
    previous = Path(rt['timing_root']) / 'predictions'
    proof = read(previous / 'complete.json')
    if (proof['status'], proof['count'], proof['target_reads'], proof['source_truth_reads']) != ('FROZEN', len(ids), 0, 0):
        raise ValueError('Frozen source-only receiver timing outputs required')
    comparison = ('direct', 'r69', 'timing_protocol')
    summaries = {}
    for route, folder in core.ROUTES.items():
        observation = cache / folder
        clean = np.load(observation / 'clean.npy', mmap_mode='r', allow_pickle=False)
        for v in core.VIEWS:
            key = f'{route}_{v}'
            with np.load(previous / f'{key}.npz', allow_pickle=False) as sealed:
                existing = sealed['decision'][[proof['modes'].index(mode) for mode in comparison]]
                if not np.array_equal(sealed['ids'], ids) or existing.shape != (3, len(ids)) or existing.dtype != np.uint8 or (existing > 5).any():
                    raise ValueError('Existing source prediction physical order or output differs')
            observed = np.load(observation / f'{v}.npy', mmap_mode='r', allow_pickle=False)
            if observed.shape != clean.shape or clean.shape != (len(ids), 2, 256):
                raise ValueError('Paired source validation IQ shape differs')
            rows, fields = [], []
            for offset in range(0, core.BATCH if smoke else len(ids), core.BATCH):
                x, y = [d[offset:offset + core.BATCH] for d in (clean, observed)]
                variants, measures = oracle_readouts(y[:, 0] + 1j * y[:, 1], x[:, 0] + 1j * x[:, 1])
                fields.append(measures)
                logits = torch.stack([f(tensor(np.stack((wave.real, wave.imag), axis=1).astype(np.float32), 'cuda:0'))
                                      for wave in variants])
                if logits.shape != (3, len(x), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite prediction in isolated source timing diagnostic')
                rows.append(logits.argmax(dim=-1).tolist())
            summaries[key] = {k: core.quantiles([r[k] for r in fields]) for k in fields[0]}
            if not smoke:
                file = root / 'predictions' / f'{key}.npz'
                file.parent.mkdir(exist_ok=True)
                with file.open('xb') as stream:
                    np.savez(stream, ids=ids, decision=np.concatenate((existing, np.concatenate(rows, axis=1))).astype(np.uint8))
                print('PAIRED_SOURCE_TIMING_FROZEN', key, len(ids), flush=True)
    write(root / ('smoke.json' if smoke else 'predictions/complete.json'), dict(
        status='PASS' if smoke else 'FROZEN', count=core.BATCH if smoke else len(ids),
        modes=list(MODES), views=list(core.VIEWS), summaries=summaries, source_truth_reads=0,
        target_reads=0, paired_clean_isolated_diagnostic=True))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    try:
        for action in ('smoke', 'evaluate', 'score'):
            gpu = None if action == 'score' else core.select_gpu()
            write(state, dict(status=action.upper(), gpu=gpu), exclusive=False)
            if core.start(rt, action, 'paired', gpu, module=rt['runner'], extra_keys=('timing_root',)).wait():
                raise RuntimeError(f'Source timing diagnostic {action} failed')
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    c = read(context)
    rt = {key: c[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(project / 'releases' / f'{RUN}-release'), run_root=str(project / 'runs' / RUN),
              source_root=str(project / 'runs' / core.CACHE_RUN / 'cache'),
              parent_root=str(project / 'runs' / core.R17_RUN), timing_root=str(project / 'runs' / TIMING))
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        cmd = actions.add_parser(action)
        cmd.add_argument('runtime', type=Path)
        cmd.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.action == 'publish':
        publish(args.repo, args.output, args.context, args.ssh_config)
        return
    rt = read(args.runtime)
    if args.action == 'dispatch':
        dispatch(rt)
    elif args.action == 'score':
        core.score(rt, MODES)
    else:
        activate(rt)
        evaluate(rt, smoke=args.action == 'smoke')


if __name__ == '__main__':
    main()
