"""Benchmark one context producer and parallel source-only recovery consumers."""
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
from experiments.phase1_dura_orthogonal.benchmark_shared_differences import compare_details
from experiments.phase1_dura_orthogonal.benchmark_shared_threads import MODELS
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.parallel_source_benchmark import collect_results, encode, wait_ready, worker
from experiments.phase1_dura_orthogonal.shared_source_benchmark import classify, same_bits
from experiments.phase1_dura_orthogonal.train_source import extract_context

DTYPES = {'torch.float32': torch.float32, 'torch.float64': torch.float64}


def decode_tensor(item):
    dtype, shape, data = item
    if dtype not in DTYPES:
        raise ValueError('Unsupported source IQ dtype')
    return torch.frombuffer(bytearray(data), dtype=DTYPES[dtype]).reshape(shape)


def consumer(root, run, commit, selected, tasks, ready, start, results):
    announced = False
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        root = Path(root)
        model, dura, reader = load_fixed(root, run, commit)
        _, _, _, _, received, _ = source_inputs(root)
        with torch.no_grad():
            x = core.tensor(received[selected[:8]], 'cpu')
            anchor, direction, variance = extract_context(dura, x)
            warm = model(x, anchor, direction, variance)
            if not torch.isfinite(classify(reader, warm)).all():
                raise FloatingPointError('Finite source consumer warmup required')
        ready.put(('READY', os.getpid()))
        announced = True
        if not start.wait(120):
            raise TimeoutError('Pipeline start was not signalled')
        with torch.no_grad():
            while True:
                task = tasks.get(timeout=120)
                if task is None:
                    break
                batch, payload = task
                if len(payload) != 6:
                    raise ValueError('Six shared source tensors required')
                x, anchor, direction, variance, raw, anchor_logits = (decode_tensor(v) for v in payload)
                restored = model(x, anchor, direction, variance)
                result = encode((anchor, direction, variance, restored, raw, anchor_logits, classify(reader, restored)))
                results.put(('OK', run, batch, result))
    except BaseException:
        if not announced:
            ready.put(('ERROR', os.getpid(), traceback.format_exc()))
        results.put(('ERROR', run, traceback.format_exc()))


def collect_pipeline(results, processes, batches, timeout=120):
    expected = {(run, batch) for _, run, _ in MODELS for batch in range(batches)}
    found = {}
    deadline = time.monotonic()+timeout
    while len(found) < len(expected):
        if any(p.exitcode not in (None, 0) for p in processes):
            raise RuntimeError('Pipeline consumer failed')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Pipeline outputs did not arrive')
        try:
            item = results.get(timeout=min(1., remaining))
        except queue.Empty:
            continue
        if item[0] != 'OK':
            raise RuntimeError('Pipeline consumer error: '+repr(item[1:]))
        key = (item[1], item[2])
        if key not in expected or key in found:
            raise ValueError('Each expected model/batch must appear exactly once')
        found[key] = item[3]
    return [[found[(run, batch)] for _, run, _ in MODELS] for batch in range(batches)]


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
                raise ValueError('Complete source groups required')
            positions.extend(group[:8].tolist())
    selected = ids[np.asarray(positions)]
    batches = [core.tensor(received[selected[i:i+8]], 'cpu') for i in range(0, len(selected), 8)]
    triples = [load_fixed(args.root, run, commit) for _, run, commit in MODELS]
    for index in (1, 2):
        for left, right in ((triples[0][1], triples[index][1]), (triples[0][2], triples[index][2])):
            a, b = left.state_dict(), right.state_dict()
            if a.keys() != b.keys() or any(not same_bits(a[k], b[k]) for k in a):
                raise ValueError('Shared context and classifier require identical frozen states')
    _, dura, reader = triples[0]
    with torch.no_grad():
        anchor, direction, variance = extract_context(dura, batches[0])
        if not all(torch.isfinite(v).all() for v in (anchor, direction, variance, classify(reader, anchor))):
            raise FloatingPointError('Finite real-checkpoint source smoke required')
    args.out.mkdir(parents=True)
    print('PIPELINED_SHARED_REAL_SOURCE_SMOKE_PASS', flush=True)
    ctx = mp.get_context('spawn')
    pairs = []
    for pipeline_first in (False, True):
        bready, presults, bstart = ctx.Queue(), ctx.Queue(), ctx.Event()
        pready, output, pstart = ctx.Queue(), ctx.Queue(), ctx.Event()
        tasks = [ctx.Queue() for _ in MODELS]
        baseline = [ctx.Process(target=worker, args=(str(args.root), run, commit, selected, bready, bstart, presults))
                    for _, run, commit in MODELS]
        pipeline = [ctx.Process(target=consumer, args=(str(args.root), run, commit, selected, tasks[index], pready, pstart, output))
                    for index, (_, run, commit) in enumerate(MODELS)]
        started = []
        loaded_path = sys.path[:]
        try:
            try:
                sys.path[:] = initial_path
                for process in baseline+pipeline:
                    process.start()
                    started.append(process)
            finally:
                sys.path[:] = loaded_path
            bpids, ppids = wait_ready(bready, baseline), wait_ready(pready, pipeline)
            def do_baseline():
                began = time.perf_counter()
                bstart.set()
                rows = collect_results(presults, baseline)
                elapsed = time.perf_counter()-began
                if any(row[0] != 'OK' for row in rows):
                    raise RuntimeError('Independent source worker failed')
                by_run = {row[1]: row[3] for row in rows}
                if set(by_run) != {run for _, run, _ in MODELS}:
                    raise ValueError('All independent model outputs required')
                return by_run, elapsed
            def do_pipeline():
                began = time.perf_counter()
                pstart.set()
                with torch.no_grad():
                    for index, x in enumerate(batches):
                        anchor, direction, variance = extract_context(dura, x)
                        payload = encode((x, anchor, direction, variance, classify(reader, x), classify(reader, anchor)))
                        for task in tasks:
                            task.put((index, payload))
                for task in tasks:
                    task.put(None)
                rows = collect_pipeline(output, pipeline, len(batches))
                return rows, time.perf_counter()-began
            if pipeline_first:
                common, pipeline_seconds = do_pipeline()
                by_run, baseline_seconds = do_baseline()
            else:
                by_run, baseline_seconds = do_baseline()
                common, pipeline_seconds = do_pipeline()
            details = compare_details(common, by_run)
            pairs.append(dict(pipeline_first=pipeline_first, independent_seconds=baseline_seconds,
                              pipeline_seconds=pipeline_seconds, baseline_pids=bpids, pipeline_pids=ppids,
                              fields=details))
            print('PIPELINED_SHARED_PAIR', pipeline_first, flush=True)
        finally:
            for process in started:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            for task in tasks:
                task.cancel_join_thread()
            for q in (bready, presults, pready, output, *tasks):
                q.close()
    core.write(args.out/'result.json', dict(status='PIPELINED_SHARED_COMPLETE', commit=args.commit,
        records=len(selected), batch_size=8, models=[name for name, _, _ in MODELS], pairs=pairs,
        independent_workers=3, independent_threads=2, producer_threads=2, consumer_workers=3, consumer_threads=2,
        TX_truth_reads=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Two short source CPU pairs, unequal total thread budgets, loading and process start excluded; not full-test parity.'))
    print('PIPELINED_SHARED_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
