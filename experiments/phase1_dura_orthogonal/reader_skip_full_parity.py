"""Compare every fixed Phase1 decision with a reader-skip replay, without truth."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import SOURCE_PILOT, source_inputs
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.source_reader_skip_probe import run_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--ground', required=True)
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique parity output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, dura, reader = load_fixed(args.root, args.source, args.candidate_commit)
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    selected = []
    for receivers in ((4, 6), (8,)):
        for view in core.VIEWS:
            match = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(match) < 8:
                raise ValueError('Eight source records per receiver group and view required')
            selected.extend(match[:8].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    with np.load(args.root/'runs'/SOURCE_PILOT/'source_predictions.npz', allow_pickle=False) as z:
        expected = z['values'][selected][:, [3, 5]].astype(np.int64)
    for start in range(0, len(selected), 8):
        x = core.tensor(received[ids[selected[start:start+8]]], 'cpu')
        standard = run_batch(x, model, dura, reader, True)
        fast = run_batch(x, model, dura, reader, False)
        if not torch.equal(standard[0], fast[0]) or not torch.equal(standard[2], fast[2]):
            raise ValueError('Source raw/restored logits differ')
        actual = np.from_dlpack(torch.stack((standard[0].argmax(1), standard[1].argmax(1)), 1).cpu().contiguous())
        if not np.array_equal(actual, expected[start:start+len(x)]):
            raise ValueError('Fixed source raw/anchor pilot mismatch')
    args.out.mkdir(parents=True)
    core.write(args.out/'source_smoke.json', dict(status='PASS', count=len(selected),
        raw_and_restored_logits_elementwise_equal=True, fixed_pilot_mismatches=0, target_reads=0, TX_truth_reads=0))
    print('READER_SKIP_SOURCE_SMOKE_PASS', len(selected), flush=True)

    folder, ids = core.reference(args.root)
    reference = args.root/'runs'/args.ground/'flow/test_prediction'
    manifest = core.read(reference/'complete.json')
    core.validate_prediction_manifest(manifest, count=len(ids), commit=args.candidate_commit)
    if not np.array_equal(ids, np.load(reference/'physical_ids.npy', allow_pickle=False)):
        raise ValueError('Complete baseline predictions with matching physical IDs required')
    started = time.monotonic()
    results = {}
    for view in core.VIEWS:
        view_started = time.monotonic()
        values = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        raw_reference = np.load(reference/(view+'.raw.npy'), mmap_mode='r', allow_pickle=False)
        restored_reference = np.load(reference/(view+'.restored.npy'), mmap_mode='r', allow_pickle=False)
        if (values.shape != (len(ids), 2, 256) or values.dtype != np.float32 or
                raw_reference.shape != (len(ids),) or restored_reference.shape != (len(ids),)):
            raise ValueError('Complete fixed input and reference arrays required')
        for start in range(0, len(ids), 8):
            x = core.tensor(values[start:start+8], 'cpu')
            raw, _, restored = run_batch(x, model, dura, reader, False)
            end = start + len(x)
            actual_raw = np.from_dlpack(raw.argmax(1).cpu().contiguous())
            actual_restored = np.from_dlpack(restored.argmax(1).cpu().contiguous())
            if not np.array_equal(actual_raw, raw_reference[start:end]):
                raise ValueError('Raw decision mismatch at '+view+' '+str(start))
            if not np.array_equal(actual_restored, restored_reference[start:end]):
                raise ValueError('Restored decision mismatch at '+view+' '+str(start))
            if end % 16384 == 0:
                print('READER_SKIP_PROGRESS', view, end, round(time.monotonic()-started, 1), flush=True)
        results[view] = dict(count=len(ids), raw_mismatches=0, restored_mismatches=0,
                             elapsed_seconds=time.monotonic()-view_started)
        print('READER_SKIP_VIEW_COMPLETE', view, len(ids), flush=True)
    core.write(args.out/'result.json', dict(status='FULL_DECISION_PARITY_PASS', commit=args.commit,
        candidate=args.source, candidate_commit=args.candidate_commit, ground_reference=args.ground,
        F_run=core.F_RUN, F_row=core.F_ROW, batch_size=8, threads=2, views=results,
        elapsed_seconds=time.monotonic()-started, TX_truth_reads=0, model_updates=0,
        scope='raw and restored class decisions only; ground logits not stored in reference'))
    print('READER_SKIP_FULL_PARITY_PASS', flush=True)


if __name__ == '__main__':
    main()
