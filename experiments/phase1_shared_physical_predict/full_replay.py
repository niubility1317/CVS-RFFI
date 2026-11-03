"""After original predictions finish, verify every saved decision without truth."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.phase1_dura_physical_fusion.pipeline import load_components
from experiments.phase1_shared_physical_predict.parallel_benchmark import configure, dependencies, model_for
from experiments.phase1_shared_physical_predict.shared import predict_shared
from experiments.phase1_shared_physical_predict.source_benchmark import SOURCES

GROUND = (
    '20261103-phase1-dura-physical-fusion-ground-test-s2026092701-r02',
    '20261103-phase1-dura-physical-wide-ground-test-s2026092701-r01',
    '20261103-phase1-dura-physical-local-teacher-ground-test-s2026092701-r01',
)


def check_decisions(actual, expected_raw, expected_restored):
    """Require exactly equal class decisions for a matching input batch."""
    raw, restored = actual
    if not np.array_equal(raw, expected_raw) or not np.array_equal(restored, expected_restored):
        raise ValueError('Shared decisions differ from fixed original predictions')


def original_complete(root, run, commit, views):
    destination = root / 'runs' / run / 'flow' / 'test_prediction'
    complete = destination / 'complete.json'
    if not complete.exists():
        state = root / 'runs' / (run + '-controller') / 'status.json'
        if state.exists() and json.loads(state.read_text(encoding='utf8')).get('state') == 'FAILED':
            raise RuntimeError('Original prediction controller failed: ' + run)
        return None
    info = json.loads(complete.read_text(encoding='utf8'))
    expected = dict(status='PREDICTIONS_COMPLETE', count=168000, commit=commit,
                    TX_truth_reads=0, model_updates=0)
    if any(info.get(k) != v for k, v in expected.items()) or info.get('views') != list(views):
        raise ValueError('Complete original seven-view predictions required: ' + run)
    if set(info.get('accepted_by_view', {})) != set(views):
        raise ValueError('Complete original gate counts required: ' + run)
    return destination, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique replay output required')
    args.out.mkdir(parents=True)
    smoke = subprocess.run([sys.executable, '-X', 'utf8', '-u', '-m',
                            'experiments.phase1_shared_physical_predict.source_benchmark',
                            '--root', str(args.root), '--out', str(args.out / 'source_smoke')], check=False)
    if smoke.returncode:
        raise RuntimeError('Real-checkpoint source replay failed')
    configure()
    deps = dependencies()
    while True:
        originals = [original_complete(args.root, run, source[1], deps.core.VIEWS)
                     for run, source in zip(GROUND, SOURCES)]
        if all(item is not None for item in originals):
            break
        print('WAITING_ORIGINAL_COMPLETE', sum(item is not None for item in originals), '/ 3', flush=True)
        time.sleep(60)
    while os.getloadavg()[0] >= 72:
        print('WAITING_CPU', round(os.getloadavg()[0], 1), flush=True)
        time.sleep(60)
    if any(info.get('F_run') != deps.core.F_RUN or info.get('F_row') != deps.core.F_ROW
           for _, info in originals):
        raise ValueError('Original predictions must use the same frozen Phase1 reader')

    models = [model_for(args.root, i) for i in range(3)]
    _, gate, dura, physical = load_components(args.root, 'cpu', deps)
    reader, _ = deps.core.teacher(args.root, 'cpu')
    reader.eval().requires_grad_(False)
    folder, ids = deps.core.reference(args.root)
    if len(ids) != 168000:
        raise ValueError('Complete 168000-record Phase1 input required')
    for destination, _ in originals:
        if not np.array_equal(ids, np.load(destination / 'physical_ids.npy', allow_pickle=False)):
            raise ValueError('Original and replay physical IDs must match')

    started = time.monotonic()
    results = {}
    with torch.no_grad():
        for view in deps.core.VIEWS:
            values = np.load(folder / (view + '.npy'), mmap_mode='r', allow_pickle=False)
            if values.shape != (len(ids), 2, 256) or values.dtype != np.float32:
                raise ValueError('Complete fixed Phase1 inputs required')
            files = [(np.load(d / (view + '.raw.npy'), mmap_mode='r', allow_pickle=False),
                      np.load(d / (view + '.restored.npy'), mmap_mode='r', allow_pickle=False))
                     for d, _ in originals]
            if any(raw.shape != (len(ids),) or restored.shape != (len(ids),) or
                   raw.dtype != np.int64 or restored.dtype != np.int64 for raw, restored in files):
                raise ValueError('Complete fixed original decision arrays required')
            accepted = [0, 0, 0]
            for start in range(0, len(ids), 8):
                x = deps.core.tensor(values[start:start+8], 'cpu')
                output = predict_shared(x, models, gate, dura, physical, reader)
                end = start + len(x)
                for index, (_, _, active, classes) in enumerate(output):
                    raw = np.from_dlpack(classes[0].cpu().contiguous())
                    restored = np.from_dlpack(classes[2].cpu().contiguous())
                    check_decisions((raw, restored), files[index][0][start:end], files[index][1][start:end])
                    accepted[index] += int(active.sum())
                if end % 16384 == 0:
                    print('FULL_REPLAY_PROGRESS', view, end, round(time.monotonic()-started, 1), flush=True)
            for index, (_, info) in enumerate(originals):
                if accepted[index] != info['accepted_by_view'][view]:
                    raise ValueError('Shared gate count differs from fixed original: ' + view)
            results[view] = dict(records=len(ids), prediction_mismatches=0, gate_counts=accepted)
            print('FULL_REPLAY_VIEW_COMPLETE', view, len(ids), flush=True)
    result = dict(status='PASS', views=results, candidate_count=3, batch_size=8,
                  target_records_per_candidate=len(ids)*len(deps.core.VIEWS), TX_truth_reads=0,
                  model_updates=0, original_outputs_modified=False,
                  scope='Exact raw and restored class decisions and per-view gate counts; original full waveforms were not saved.')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('FULL_REPLAY_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
