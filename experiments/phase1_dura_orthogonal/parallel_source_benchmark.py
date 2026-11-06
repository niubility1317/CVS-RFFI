"""Compare three parallel frozen forwards to shared inference on source IQ."""
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.shared_source_benchmark import MODELS, independent, shared, compare


def encode(outputs):
    """Return exact tensor contents for cross-process comparisons."""
    return tuple((str(t.dtype), tuple(t.shape), t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                 for t in outputs)


def worker(root, run, commit, selected, ready, start, results):
    announced = False
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        root = Path(root)
        _, _, _, _, received, _ = source_inputs(root)
        triple = load_fixed(root, run, commit)
        batches = [core.tensor(received[selected[i:i+8]], 'cpu') for i in range(0, len(selected), 8)]
        independent(batches[0], [triple])
        ready.put(('READY', os.getpid()))
        announced = True
        if not start.wait(120):
            raise TimeoutError('Shared start was not signalled')
        began = time.perf_counter()
        outputs = [encode(independent(x, [triple])[0]) for x in batches]
        results.put(('OK', run, time.perf_counter()-began, outputs))
    except BaseException:
        if not announced:
            ready.put(('ERROR', os.getpid(), traceback.format_exc()))
        results.put(('ERROR', run, traceback.format_exc()))


def wait_ready(ready, processes, timeout=120):
    deadline = time.monotonic()+timeout
    received = []
    while len(received) < len(processes):
        if any(p.exitcode is not None for p in processes):
            raise RuntimeError('Parallel worker exited before start')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Parallel workers did not become ready')
        try:
            item = ready.get(timeout=min(1., remaining))
        except queue.Empty:
            continue
        if item[0] != 'READY':
            raise RuntimeError('Parallel worker preflight failed: '+repr(item))
        received.append(item)
    return [item[1] for item in received]


def collect_results(results, processes, timeout=120):
    deadline = time.monotonic()+timeout
    received = []
    while len(received) < len(processes):
        if any(p.exitcode not in (None, 0) for p in processes):
            raise RuntimeError('Parallel worker failed during inference')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Parallel workers did not return results')
        try:
            received.append(results.get(timeout=min(1., remaining)))
        except queue.Empty:
            continue
    return received


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    initial_path = sys.path[:]
    if args.out.exists():
        raise FileExistsError('Unique output root required')
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
    compare(independent(batches[0], triples), shared(batches[0], triples))
    args.out.mkdir(parents=True)
    print('PARALLEL_SOURCE_SMOKE_PASS', flush=True)
    ctx = mp.get_context('spawn')
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    processes = [ctx.Process(target=worker, args=(str(args.root), run, commit, selected, ready, start, results))
                 for _, run, commit in MODELS]
    loaded_path = sys.path[:]
    started_processes = []
    try:
        try:
            sys.path[:] = initial_path
            for p in processes:
                p.start()
                started_processes.append(p)
        finally:
            sys.path[:] = loaded_path
        worker_pids = wait_ready(ready, processes)
        started = time.perf_counter()
        start.set()
        parallel_outputs = collect_results(results, processes)
        parallel_seconds = time.perf_counter()-started
        if any(item[0] != 'OK' for item in parallel_outputs):
            raise RuntimeError('Parallel worker failed: '+repr([x[:3] for x in parallel_outputs if x[0] != 'OK']))
        by_run = {item[1]: item[3] for item in parallel_outputs}
        if len(by_run) != len(MODELS):
            raise ValueError('All distinct model outputs are required')
        shared_started = time.perf_counter()
        common = [[encode(values) for values in shared(x, triples)] for x in batches]
        shared_seconds = time.perf_counter()-shared_started
        for batch_index, items in enumerate(common):
            for model_index, (_, run, _) in enumerate(MODELS):
                if items[model_index] != by_run[run][batch_index]:
                    raise ValueError('Parallel and shared source outputs differ at batch '+str(batch_index))
        if shared_seconds <= 0:
            raise ValueError('Positive timing required')
        core.write(args.out/'result.json', dict(status='PARALLEL_SOURCE_BENCHMARK_COMPLETE', commit=args.commit,
            records=len(selected), batch_size=8, models=[name for name, _, _ in MODELS], worker_pids=worker_pids,
            independent_parallel_seconds=parallel_seconds, shared_seconds=shared_seconds,
            speedup=parallel_seconds/shared_seconds, per_worker_seconds={item[1]: item[2] for item in parallel_outputs},
            exact_tensor_mismatches=0, exact_scope='112 source records: DURA context, restored IQ and classifier logits for three models',
            source_only=True, TX_truth_reads=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0,
            model_updates=0, caveat='One source CPU timing, excludes model loading and process startup; includes output serialization and parallel result transfer. Not target-wide equivalence.'))
        print('PARALLEL_SOURCE_BENCHMARK_COMPLETE', flush=True)
    finally:
        for process in started_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        ready.close()
        results.close()


if __name__ == '__main__':
    main()
