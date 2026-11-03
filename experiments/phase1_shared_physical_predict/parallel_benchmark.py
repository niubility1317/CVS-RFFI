"""Compare three concurrent original predictors with one shared predictor on source IQ."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from experiments.phase1_dura_physical_fusion.pipeline import load_components, predict_batch
from experiments.phase1_shared_physical_predict.shared import predict_shared
from experiments.phase1_shared_physical_predict.source_benchmark import SOURCES


def dependencies():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase1_joint_mixture'))
    import source_pilot
    return source_pilot


def configure():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def model_for(root, index):
    name, commit, schema, factory = SOURCES[index]
    folder = root / 'runs' / name
    meta = json.loads((folder / 'prediction_complete.json').read_text(encoding='utf8'))
    required = dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit, count=672,
                    fit_ids=8192, F_frozen=True, blind_frozen=True, gate_frozen=True,
                    DURA_loaded=True, DURA_frozen=True, TX_labels_read=0,
                    U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0)
    if any(meta.get(k) != v for k, v in required.items()):
        raise ValueError('Matching frozen source prediction metadata required')
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if checkpoint.get('schema') != schema or checkpoint.get('commit') != commit or checkpoint.get('epochs') != 5:
        raise ValueError('Matching frozen checkpoint required')
    model = factory()
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False)


def flatten_batches(batches):
    """Copy outputs into arrays owned by the current NumPy installation."""
    fields = [[], [], [], [], [], []]
    for anchor, output, active, classes in batches:
        for field, value in zip(fields, (anchor, output, active, *classes)):
            field.append(value.detach().cpu())
    return [np.from_dlpack(torch.cat(field).contiguous()).copy() for field in fields]


def worker(root, index, values, ready, start, results):
    try:
        configure()
        deps = dependencies()
        model = model_for(root, index)
        blind, gate, dura, physical = load_components(root, 'cpu', deps)
        reader, _ = deps.core.teacher(root, 'cpu')
        reader.eval().requires_grad_(False)
        with torch.no_grad():
            predict_batch(deps.core.tensor(values[:8], 'cpu'), model, blind, gate, dura, physical, reader, deps)
            ready.put((index, None))
            if not start.wait(180):
                raise TimeoutError('Concurrent benchmark start timed out')
            tick = time.monotonic()
            batches = [predict_batch(deps.core.tensor(values[i:i+8], 'cpu'), model, blind, gate, dura, physical, reader, deps)
                       for i in range(8, len(values), 8)]
            finished = time.monotonic()
            elapsed = finished - tick
        results.put((index, elapsed, finished, flatten_batches(batches), None))
    except Exception:
        error = traceback.format_exc()
        ready.put((index, error))
        results.put((index, None, None, None, error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique benchmark output required')
    args.out.mkdir(parents=True)
    configure()
    deps = dependencies()
    pilots = []
    for name, _, _, _ in SOURCES:
        with np.load(args.root / 'runs' / name / 'source_predictions.npz', allow_pickle=False) as z:
            pilots.append({k: z[k].copy() for k in ('indices', 'rx', 'scene', 'values')})
    for pilot in pilots[1:]:
        if any(not np.array_equal(pilot[k], pilots[0][k]) for k in ('indices', 'rx', 'scene')):
            raise ValueError('Identical frozen source records required')
    positions = []
    for receivers in ((4, 6), (8,)):
        for view in deps.core.VIEWS:
            matches = np.flatnonzero(np.isin(pilots[0]['rx'], receivers) & (pilots[0]['scene'] == view))
            if len(matches) < 8:
                raise ValueError('Eight records per source receiver group and view required')
            positions.extend(matches[:8].tolist())
    positions = np.asarray(positions, dtype=np.int64)
    ids = pilots[0]['indices'][positions]
    cache, _, rx, _, _, _ = deps.s.source_contract(args.root)
    received, scenes = deps.s.epoch_data(cache, 19)
    if not np.array_equal(rx[ids], pilots[0]['rx'][positions]) or not np.array_equal(scenes[ids], pilots[0]['scene'][positions]):
        raise ValueError('Source IDs and scenes must match the frozen pilots')
    values = np.asarray(received[ids]).copy()

    ctx = mp.get_context('spawn')
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=worker, args=(args.root, i, values, ready, start, results)) for i in range(3)]
    for process in workers:
        process.start()
    try:
        for _ in workers:
            _, error = ready.get(timeout=180)
            if error:
                raise RuntimeError(error)
        tick = time.monotonic()
        start.set()
        outputs = {}
        for _ in workers:
            index, elapsed, finished, arrays, error = results.get(timeout=300)
            if error:
                raise RuntimeError(error)
            outputs[index] = (elapsed, arrays, finished)
        parallel_wall = max(value[2] for value in outputs.values()) - tick
        for process in workers:
            process.join(timeout=20)
            if process.exitcode != 0:
                raise RuntimeError('Concurrent predictor failed')
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join()

    models = [model_for(args.root, i) for i in range(3)]
    _, gate, dura, physical = load_components(args.root, 'cpu', deps)
    reader, _ = deps.core.teacher(args.root, 'cpu')
    reader.eval().requires_grad_(False)
    with torch.no_grad():
        predict_shared(deps.core.tensor(values[:8], 'cpu'), models, gate, dura, physical, reader)
        tick = time.monotonic()
        all_batches = [predict_shared(deps.core.tensor(values[i:i+8], 'cpu'), models, gate, dura, physical, reader)
                       for i in range(8, len(values), 8)]
        shared_wall = time.monotonic() - tick
    for i in range(3):
        actual = flatten_batches([batch[i] for batch in all_batches])
        if any(not np.array_equal(a, b) for a, b in zip(actual, outputs[i][1])):
            raise ValueError('Shared and concurrent outputs differ')
        decisions = np.column_stack((actual[2].astype(np.int64), actual[3], actual[4], actual[5]))
        if not np.array_equal(decisions, pilots[i]['values'][positions[8:]][:, [2, 3, 5, 7]].astype(np.int64)):
            raise ValueError('Results differ from frozen source pilot')
    result = dict(status='PASS', source_only=True, tested_records=104, warmup_records=8,
                  parallel_original_wall_seconds=parallel_wall,
                  parallel_original_worker_seconds=[outputs[i][0] for i in range(3)],
                  shared_wall_seconds=shared_wall, wall_speedup=parallel_wall/shared_wall,
                  exact_waveform_gate_and_class_equality=True, TX_truth_reads=0, target_IQ_reads=0,
                  caveat='One source sample and one sequential ordering under shared CPU load; full target and end-to-end handover are not verified.')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
