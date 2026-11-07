"""Compare equal-thread-budget parallel and shared inference on source IQ."""
import argparse
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.parallel_source_benchmark import (
    collect_results, encode, wait_ready, worker)
from experiments.phase1_dura_orthogonal.shared_source_benchmark import same_bits, shared

MODELS = (
    ('envelope', '20261107-phase1-dura-monotone-envelope-source-s2026092701-r01',
     'ba3a01a6267cec4caa7d4d52e0d5c640f7ddee9c'),
    ('flow', '20261107-phase1-dura-iqflow-source-s2026092701-r01',
     '8ac105153dc9ec337e82346ee0f72b97b93d8bf7'),
    ('response', '20261107-phase1-dura-counterfactual-response-source-s2026092701-r01',
     '2eadeeed2f24e22fe69c516549fe5272fca88c22'),
)


def compare_encoded(common, by_run):
    """Count mismatched batch/model tuples; each tuple contains all seven outputs."""
    return sum(items[i] != by_run[run][batch] for batch, items in enumerate(common)
               for i, (_, run, _) in enumerate(MODELS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source-only benchmark output required')
    initial_path = sys.path[:]
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    positions = []
    for receivers in ((4, 6), (8,)):
        for view in core.VIEWS:
            group = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(group) != 48:
                raise ValueError('Complete source cells required')
            positions.extend(group[:8].tolist())
    selected = ids[np.asarray(positions)]
    batches = [core.tensor(received[selected[i:i+8]], 'cpu') for i in range(0, len(selected), 8)]
    triples = [load_fixed(args.root, run, commit) for _, run, commit in MODELS]
    for index in (1, 2):
        for left, right in ((triples[0][1], triples[index][1]), (triples[0][2], triples[index][2])):
            a, b = left.state_dict(), right.state_dict()
            if a.keys() != b.keys() or any(not same_bits(a[k], b[k]) for k in a):
                raise ValueError('Shared models require identical frozen DURA and classifier states')
    with torch.no_grad():
        shared(batches[0], triples)
    args.out.mkdir(parents=True)
    print('SHARED_THREADS_REAL_SOURCE_SMOKE_PASS', flush=True)
    ctx = mp.get_context('spawn')
    pairs = []
    for shared_first in (False, True):
        ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
        processes = [ctx.Process(target=worker, args=(str(args.root), run, commit, selected, ready, start, results))
                     for _, run, commit in MODELS]
        loaded_path = sys.path[:]
        started = []
        try:
            try:
                sys.path[:] = initial_path
                for process in processes:
                    process.start()
                    started.append(process)
            finally:
                sys.path[:] = loaded_path
            pids = wait_ready(ready, processes)
            def do_parallel():
                began = time.perf_counter()
                start.set()
                output = collect_results(results, processes)
                elapsed = time.perf_counter()-began
                if any(item[0] != 'OK' for item in output):
                    raise RuntimeError('Parallel source worker failed')
                by_run = {item[1]: item[3] for item in output}
                if set(by_run) != {run for _, run, _ in MODELS}:
                    raise ValueError('All distinct model outputs required')
                return by_run, elapsed
            def do_shared():
                began = time.perf_counter()
                with torch.no_grad():
                    common = [[encode(values) for values in shared(x, triples)] for x in batches]
                return common, time.perf_counter()-began
            if shared_first:
                common, shared_seconds = do_shared()
                by_run, parallel_seconds = do_parallel()
            else:
                by_run, parallel_seconds = do_parallel()
                common, shared_seconds = do_shared()
            mismatches = compare_encoded(common, by_run)
            pairs.append(dict(shared_first=shared_first, parallel_seconds=parallel_seconds,
                              shared_seconds=shared_seconds, worker_pids=pids,
                              mismatched_batch_model_tuples=mismatches))
            if mismatches:
                break
        finally:
            for process in started:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            ready.close()
            results.close()
    core.write(args.out/'result.json', dict(
        status='SHARED_THREADS_COMPLETE' if len(pairs) == 2 and all(not p['mismatched_batch_model_tuples'] for p in pairs)
        else 'SHARED_THREADS_MISMATCH', commit=args.commit, records=len(selected), batch_size=8,
        models=[name for name, _, _ in MODELS], independent_workers=3, threads_per_worker=2,
        shared_workers=1, shared_threads=6, pairs=pairs, source_only=True,
        TX_truth_reads=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Two short source CPU pairs; model loading and process startup excluded; parallel includes result transfer. No target-wide equivalence.'))
    print('SHARED_THREADS_COMPLETE' if len(pairs) == 2 and all(not p['mismatched_batch_model_tuples'] for p in pairs)
          else 'SHARED_THREADS_MISMATCH', flush=True)


if __name__ == '__main__':
    main()
