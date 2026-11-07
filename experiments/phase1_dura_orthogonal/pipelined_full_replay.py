"""Truth-free full Phase1 replay of shared pipeline versus three fixed models."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.benchmark_pipelined_shared import collect_pipeline, consumer, decode_tensor
from experiments.phase1_dura_orthogonal.benchmark_pipelined_balanced import GROUPS, pooled_consumer, validate_groups
from experiments.phase1_dura_orthogonal.benchmark_shared_differences import FIELDS, compare_details, decode
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.parallel_source_benchmark import encode, wait_ready
from experiments.phase1_dura_orthogonal.shared_source_benchmark import MODELS, classify, independent, same_bits
from experiments.phase1_dura_orthogonal.train_source import extract_context

GROUNDS = {
    '20261106-phase1-dura-orthogonal-source-s2026092701-r01': '20261106-phase1-dura-orthogonal-ground-test-s2026092701-r01',
    '20261106-phase1-dura-orthogonal-guard-source-s2026092701-r01': '20261106-phase1-dura-orthogonal-guard-ground-test-s2026092701-r01',
    '20261106-phase1-dura-band-guard-source-s2026092701-r01': '20261106-phase1-dura-band-guard-ground-test-s2026092701-r01',
}
CHUNK_BATCHES = 128


def baseline_consumer(root, run, commit, selected, tasks, ready, start, results):
    announced = False
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        root = Path(root)
        triple = load_fixed(root, run, commit)
        _, _, _, _, received, _ = source_inputs(root)
        with torch.no_grad():
            warm = independent(core.tensor(received[selected[:8]], 'cpu'), [triple])[0]
            if not all(torch.isfinite(v).all() for v in warm):
                raise FloatingPointError('Finite independent source warmup required')
        ready.put(('READY', os.getpid()))
        announced = True
        if not start.wait(120):
            raise TimeoutError('Replay start was not signalled')
        with torch.no_grad():
            while True:
                task = tasks.get(timeout=300)
                if task is None:
                    break
                batch, payload = task
                result = independent(decode_tensor(payload), [triple])[0]
                results.put(('OK', run, batch, encode(result)))
    except BaseException:
        if not announced:
            ready.put(('ERROR', os.getpid(), traceback.format_exc()))
        results.put(('ERROR', run, traceback.format_exc()))


def decision_mismatches(by_model, expected, batch_size=8):
    """Compare replay logits to prior fixed raw/restored decisions, no truth."""
    if len(by_model) != len(expected):
        raise ValueError('One expected pair per model required')
    changes = []
    for index, (raw_expected, restored_expected) in enumerate(expected):
        raw, restored = [], []
        for row in by_model[index]:
            raw.extend(decode(row[4]).argmax(1).tolist())
            restored.extend(decode(row[6]).argmax(1).tolist())
        if len(raw) != len(raw_expected) or len(restored) != len(restored_expected):
            raise ValueError('Matching saved prediction lengths required')
        changes.append(dict(raw=int(np.count_nonzero(np.asarray(raw) != raw_expected)),
                            restored=int(np.count_nonzero(np.asarray(restored) != restored_expected))))
    return changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--pooled-two', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique replay output required')
    initial_path = sys.path[:]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_ids, _, _, _, received, _ = source_inputs(args.root)
    triples = [load_fixed(args.root, run, commit) for _, run, commit in MODELS]
    for index in (1, 2):
        for left, right in ((triples[0][1], triples[index][1]), (triples[0][2], triples[index][2])):
            a, b = left.state_dict(), right.state_dict()
            if a.keys() != b.keys() or any(not same_bits(a[k], b[k]) for k in a):
                raise ValueError('Identical fixed DURA and classifier state required')
    if any(p.requires_grad for triple in triples for model in triple for p in model.parameters()):
        raise ValueError('Frozen models required')
    _, dura, reader = triples[0]
    ctx = mp.get_context('spawn')
    bready, pready, boutput, poutput = (ctx.Queue() for _ in range(4))
    bstart, pstart = ctx.Event(), ctx.Event()
    btasks = [ctx.Queue() for _ in MODELS]
    if args.pooled_two:
        validate_groups(GROUPS, MODELS)
    ptasks = [ctx.Queue() for _ in (GROUPS if args.pooled_two else MODELS)]
    baseline = [ctx.Process(target=baseline_consumer, args=(str(args.root), run, commit, source_ids,
                          btasks[i], bready, bstart, boutput)) for i, (_, run, commit) in enumerate(MODELS)]
    if args.pooled_two:
        pipeline = [ctx.Process(target=pooled_consumer, args=(str(args.root), tuple(MODELS[i] for i in indices),
                              source_ids, ptasks[index], pready, pstart, poutput)) for index, indices in enumerate(GROUPS)]
    else:
        pipeline = [ctx.Process(target=consumer, args=(str(args.root), run, commit, source_ids,
                              ptasks[i], pready, pstart, poutput)) for i, (_, run, commit) in enumerate(MODELS)]
    started = []
    loaded_path = sys.path[:]
    args.out.mkdir(parents=True)
    state = dict(status='RUNNING', commit=args.commit, views={}, TX_truth_reads=0, model_updates=0,
                 models=[run for _, run, _ in MODELS], independent_threads=6,
                 pipeline_threads=2+2*len(pipeline), pooled_two=args.pooled_two)
    def save():
        temp = args.out/'progress.tmp'
        temp.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n', encoding='utf8')
        os.replace(temp, args.out/'progress.json')
    def do_baseline(batches):
        began = time.perf_counter()
        for i, x in enumerate(batches):
            payload = encode((x,))[0]
            for tasks in btasks:
                tasks.put((i, payload))
        rows = collect_pipeline(boutput, baseline, len(batches), timeout=300, models=MODELS)
        return rows, time.perf_counter()-began
    def do_pipeline(batches):
        began = time.perf_counter()
        with torch.no_grad():
            for i, x in enumerate(batches):
                anchor, direction, variance = extract_context(dura, x)
                payload = encode((x, anchor, direction, variance, classify(reader, x), classify(reader, anchor)))
                for tasks in ptasks:
                    tasks.put((i, payload))
        rows = collect_pipeline(poutput, pipeline, len(batches), timeout=300, models=MODELS)
        return rows, time.perf_counter()-began
    def compare(a, b):
        by_run = {run: [row[i] for row in a] for i, (_, run, _) in enumerate(MODELS)}
        return compare_details(b, by_run, models=MODELS)
    try:
        try:
            sys.path[:] = initial_path
            for process in baseline+pipeline:
                process.start()
                started.append(process)
        finally:
            sys.path[:] = loaded_path
        state['baseline_pids'] = wait_ready(bready, baseline)
        state['pipeline_pids'] = wait_ready(pready, pipeline)
        bstart.set(); pstart.set()
        smoke = [core.tensor(received[source_ids[:8]], 'cpu')]
        a, _ = do_baseline(smoke)
        b, _ = do_pipeline(smoke)
        if any(v['mismatched'] for v in compare(a, b).values()):
            raise ValueError('Real source smoke requires exact seven-tensor agreement')
        print('PIPELINED_FULL_REAL_SOURCE_SMOKE_PASS', flush=True)
        folder, ids = core.reference(args.root)
        fixed = {}
        for _, run, commit in MODELS:
            location = args.root/'runs'/GROUNDS[run]/'flow/test_prediction'
            done = core.read(location/'complete.json')
            core.validate_prediction_manifest(done, count=168000, commit=commit)
            if done.get('views') != list(core.VIEWS) or not np.array_equal(ids, np.load(location/'physical_ids.npy', allow_pickle=False)):
                raise ValueError('Previous fixed seven-view predictions must share physical IDs')
            fixed[run] = location
        save()
        chunk_index = 0
        for view in core.VIEWS:
            iq = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
            expected = [(np.load(fixed[run]/(view+'.raw.npy'), mmap_mode='r', allow_pickle=False),
                         np.load(fixed[run]/(view+'.restored.npy'), mmap_mode='r', allow_pickle=False))
                        for _, run, _ in MODELS]
            if iq.shape != (len(ids), 2, 256) or iq.dtype != np.float32 or any(
                    x.shape != ids.shape or x.dtype != np.int64 for pair in expected for x in pair):
                raise ValueError('Complete IQ and previous predictions required')
            info = dict(processed=0, fields={k: dict(compared=0, mismatched=0, max_abs=0., class_changes=0) for k in FIELDS},
                        prior_decision_mismatches={run: dict(raw=0, restored=0) for _, run, _ in MODELS},
                        independent_seconds=0., pipeline_seconds=0.)
            state['views'][view] = info
            for begin in range(0, len(ids), 8*CHUNK_BATCHES):
                end = min(begin+8*CHUNK_BATCHES, len(ids))
                batches = [core.tensor(iq[i:min(i+8, end)], 'cpu') for i in range(begin, end, 8)]
                if chunk_index % 2:
                    b, pt = do_pipeline(batches); a, bt = do_baseline(batches)
                else:
                    a, bt = do_baseline(batches); b, pt = do_pipeline(batches)
                details = compare(a, b)
                per_model = [[row[i] for row in a] for i in range(len(MODELS))]
                prior = decision_mismatches(per_model, [(x[begin:end], y[begin:end]) for x, y in expected])
                for field, values in details.items():
                    target = info['fields'][field]
                    for key in ('compared', 'mismatched', 'class_changes'):
                        target[key] += values[key]
                    target['max_abs'] = max(target['max_abs'], values['max_abs'])
                for (_, run, _), change in zip(MODELS, prior):
                    for key in ('raw', 'restored'):
                        info['prior_decision_mismatches'][run][key] += change[key]
                info['processed'] = end
                info['independent_seconds'] += bt
                info['pipeline_seconds'] += pt
                chunk_index += 1
                if any(v['mismatched'] for v in details.values()) or any(v['raw'] or v['restored'] for v in prior):
                    state['status'] = 'MISMATCH'
                    save()
                    raise ValueError('Full replay output or prior decisions differ')
                if end % 16384 == 0 or end == len(ids):
                    save()
                    print('PIPELINED_FULL_PROGRESS', view, end, flush=True)
            print('PIPELINED_FULL_VIEW_COMPLETE', view, len(ids), flush=True)
        state['status'] = 'FULL_REPLAY_EXACT'
        save()
        print('PIPELINED_FULL_REPLAY_EXACT', flush=True)
    finally:
        for tasks in btasks+ptasks:
            tasks.put(None)
        for process in started:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        for tasks in btasks+ptasks:
            tasks.cancel_join_thread()
        for q in (bready, pready, boutput, poutput, *btasks, *ptasks):
            q.close()


if __name__ == '__main__':
    main()
