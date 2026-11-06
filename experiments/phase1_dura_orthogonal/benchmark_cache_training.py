"""Compare one fixed source training run with and without source context reuse."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.pamf_phase1_flow import experiment as core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--both-cached', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source-only timing output required')
    args.out.mkdir(parents=True)
    if args.deterministic:
        smoke = [sys.executable, '-X', 'utf8', '-u', '-m',
                 'experiments.phase1_dura_orthogonal.train_source', '--root', str(args.root),
                 '--out', str(args.out/'smoke'), '--commit', args.commit, '--device', 'cuda:0',
                 '--prepare-batch', '32', '--objective', 'dura_guard', '--architecture', 'dura_multiscale',
                 '--deterministic', '--smoke']
        with (args.out/'smoke.log').open('x', encoding='utf8') as log:
            subprocess.run(smoke, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        smoke_result = core.read(args.out/'smoke/smoke.json')
        if smoke_result.get('status') != 'SOURCE_TRAINING_SMOKE_PASS' or not smoke_result.get('deterministic_algorithms'):
            raise ValueError('Deterministic real-checkpoint source smoke required')
        print('DETERMINISTIC_SOURCE_SMOKE_PASS', flush=True)
    times = {}
    for name in ('fresh', 'cached'):
        cmd = [sys.executable, '-X', 'utf8', '-u', '-m',
               'experiments.phase1_dura_orthogonal.train_source', '--root', str(args.root),
               '--out', str(args.out/name), '--commit', args.commit, '--device', 'cuda:0',
               '--prepare-batch', '32', '--objective', 'dura_guard', '--architecture', 'dura_multiscale']
        if name == 'cached' or args.both_cached:
            cmd += ['--source-context-cache', str(args.cache)]
        if args.deterministic:
            cmd += ['--deterministic']
        print('TRAINING_PARITY_START', name, flush=True)
        start = time.monotonic()
        with (args.out/(name+'.log')).open('x', encoding='utf8') as log:
            subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        times[name] = time.monotonic()-start
        print('TRAINING_PARITY_COMPLETE', name, round(times[name], 3), flush=True)
    fresh = torch.load(args.out/'fresh/final.pt', map_location='cpu', weights_only=False)
    cached = torch.load(args.out/'cached/final.pt', map_location='cpu', weights_only=False)
    if fresh['model'].keys() != cached['model'].keys():
        raise ValueError('Source recovery model state keys differ')
    different = [k for k in fresh['model'] if not torch.equal(fresh['model'][k], cached['model'][k])]
    maximum = max((float((fresh['model'][k]-cached['model'][k]).abs().max())
                   for k in different), default=0.)
    summaries = {name: core.read(args.out/name/'source_summary.json') for name in ('fresh', 'cached')}
    for summary in summaries.values():
        if summary.get('status') != 'SOURCE_TRAINED' or any(summary.get(k) != 0 for k in
                ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads')):
            raise ValueError('Complete source-only training results required')
    core.write(args.out/'result.json', dict(status='SOURCE_TRAINING_PARITY_COMPLETE', commit=args.commit,
        comparison='cached_vs_cached' if args.both_cached else 'fresh_vs_cached',
        deterministic_algorithms=args.deterministic,
        model_exact=not different, different_parameters=different, max_abs_parameter_difference=maximum,
        fit_ids_equal=fresh['fit_ids'].tolist() == cached['fit_ids'].tolist(),
        training_losses_exact=fresh['losses'] == cached['losses'],
        source_groups_exact=summaries['fresh']['source_groups'] == summaries['cached']['source_groups'],
        elapsed_seconds=times,
        preparation_seconds={name: summaries[name]['preparation_seconds'] for name in summaries},
        device='cuda:0', batch_size=32, source_only=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0,
        note='No target predictions were generated; exact checkpoint equality does not establish bitwise ground predictions across environments.'))
    print('TRAINING_PARITY_FINISHED', flush=True)


if __name__ == '__main__':
    main()
