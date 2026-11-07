"""Localize shared-inference numerical differences on source IQ only."""
import argparse
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.benchmark_shared_threads import MODELS
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.parallel_source_benchmark import collect_results, encode, wait_ready, worker
from experiments.phase1_dura_orthogonal.shared_source_benchmark import same_bits, shared

FIELDS = ('anchor', 'direction', 'variance', 'restored', 'raw_logits', 'anchor_logits', 'restored_logits')
DTYPES = {'torch.float32': np.float32, 'torch.float64': np.float64}


def decode(encoded):
    dtype, shape, raw = encoded
    if dtype not in DTYPES:
        raise ValueError('Unsupported source-output dtype: '+dtype)
    array = np.frombuffer(raw, dtype=DTYPES[dtype]).reshape(shape)
    if not np.isfinite(array).all():
        raise FloatingPointError('Finite frozen source outputs required')
    return array


def compare_details(common, by_run):
    """Aggregate bitwise, absolute and decision differences per tensor field."""
    stats = {key: dict(compared=0, mismatched=0, max_abs=0., class_changes=0) for key in FIELDS}
    for batch, items in enumerate(common):
        for index, (_, run, _) in enumerate(MODELS):
            reference = by_run[run][batch]
            current = items[index]
            if len(current) != len(FIELDS) or len(reference) != len(FIELDS):
                raise ValueError('All seven output tensors required')
            for name, a, b in zip(FIELDS, reference, current):
                if a[:2] != b[:2]:
                    raise ValueError('Matching tensor shape and dtype required')
                x, y = decode(a), decode(b)
                row = stats[name]
                row['compared'] += 1
                row['mismatched'] += int(a[2] != b[2])
                row['max_abs'] = max(row['max_abs'], float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))))
                if name.endswith('logits'):
                    if x.ndim != 2:
                        raise ValueError('Batched classification logits required')
                    row['class_changes'] += int(np.count_nonzero(x.argmax(1) != y.argmax(1)))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source-only benchmark output required')
    initial_path = sys.path[:]
    torch.set_num_threads(2)
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
        smoke = shared(batches[0], triples)
    if not all(torch.isfinite(v).all() for group in smoke for v in group):
        raise FloatingPointError('Finite real-checkpoint source smoke required')
    args.out.mkdir(parents=True)
    print('SHARED_DIFFERENCES_REAL_SOURCE_SMOKE_PASS', flush=True)
    context = mp.get_context('spawn')
    results = []
    for threads in (2, 6):
        torch.set_num_threads(threads)
        for shared_first in (False, True):
            ready, output, start = context.Queue(), context.Queue(), context.Event()
            processes = [context.Process(target=worker, args=(str(args.root), run, commit, selected, ready, start, output))
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
                def parallel():
                    began = time.perf_counter()
                    start.set()
                    rows = collect_results(output, processes)
                    elapsed = time.perf_counter()-began
                    if any(row[0] != 'OK' for row in rows):
                        raise RuntimeError('Parallel source worker failed')
                    by_run = {row[1]: row[3] for row in rows}
                    if set(by_run) != {run for _, run, _ in MODELS}:
                        raise ValueError('All distinct model outputs required')
                    return by_run, elapsed
                def common():
                    began = time.perf_counter()
                    with torch.no_grad():
                        rows = [[encode(values) for values in shared(x, triples)] for x in batches]
                    return rows, time.perf_counter()-began
                if shared_first:
                    common_rows, shared_seconds = common()
                    by_run, parallel_seconds = parallel()
                else:
                    by_run, parallel_seconds = parallel()
                    common_rows, shared_seconds = common()
                details = compare_details(common_rows, by_run)
                results.append(dict(shared_threads=threads, shared_first=shared_first,
                                    parallel_seconds=parallel_seconds, shared_seconds=shared_seconds,
                                    worker_pids=pids, fields=details))
                print('SHARED_DIFFERENCES_PAIR', threads, shared_first, flush=True)
            finally:
                for process in started:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                ready.close()
                output.close()
    if any(p.requires_grad or p.grad is not None for triple in triples for model in triple for p in model.parameters()):
        raise ValueError('All source benchmark models must remain frozen')
    core.write(args.out/'result.json', dict(status='SHARED_DIFFERENCES_COMPLETE', commit=args.commit,
        records=len(selected), batch_size=8, models=[name for name, _, _ in MODELS],
        independent_workers=3, threads_per_worker=2, pairs=results, source_only=True,
        TX_truth_reads=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Four short CPU pairs; model load and process startup excluded. Source-only, not full-test parity or stable throughput.'))
    print('SHARED_DIFFERENCES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
