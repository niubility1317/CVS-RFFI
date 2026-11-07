"""Truth-free full-view replay against an already fixed Phase1 prediction set."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.ground import decisions, load_fixed
from experiments.phase1_dura_orthogonal.source_pair_inference import logits

SOURCE = '20261107-phase1-dura-reversible-source-s2026092701-r01'
GROUND = '20261107-phase1-dura-reversible-ground-test-s2026092701-r01'
SOURCE_COMMIT = 'bf13e08dd7ca796e129b15baf53aad8660eb89f4'


def compare_batch(logit_pair, expected_raw, expected_restored):
    raw, restored = (np.from_dlpack(value.argmax(1).cpu().contiguous()) for value in logit_pair)
    if raw.shape != expected_raw.shape or restored.shape != expected_restored.shape:
        raise ValueError('Matching prediction batch shape required')
    return int(np.count_nonzero(raw != expected_raw)), int(np.count_nonzero(restored != expected_restored))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, dura, reader = load_fixed(args.root, SOURCE, SOURCE_COMMIT)
    models = (model, dura, reader)
    snapshots = [{key: value.detach().clone() for key, value in obj.state_dict().items()} for obj in models]
    if any(param.requires_grad for obj in models for param in obj.parameters()):
        raise ValueError('Frozen models required')
    source_ids, _, _, _, received, _ = source_inputs(args.root)
    with torch.no_grad():
        x = core.tensor(received[source_ids[:8]], 'cpu')
        original = decisions(x, model, dura, reader)
        pair = logits(x, model, dura, reader, include_anchor=False)
        if not torch.equal(original[0], pair[0].argmax(1)) or not torch.equal(original[2], pair[1].argmax(1)):
            raise ValueError('Original and pair source prediction smoke mismatch')
    print('PAIR_REPLAY_REAL_SOURCE_SMOKE_PASS', flush=True)
    previous = args.root/'runs'/GROUND/'flow/test_prediction'
    complete = core.read(previous/'complete.json')
    if (complete.get('status') != 'PREDICTIONS_COMPLETE' or complete.get('count') != 168000
            or complete.get('views') != list(core.VIEWS) or complete.get('commit') != SOURCE_COMMIT
            or complete.get('F_run') != core.F_RUN or complete.get('F_row') != core.F_ROW
            or complete.get('TX_truth_reads') != 0 or complete.get('model_updates') != 0):
        raise ValueError('Complete fixed original predictions required')
    folder, ids = core.reference(args.root)
    if not np.array_equal(ids, np.load(previous/'physical_ids.npy', allow_pickle=False)):
        raise ValueError('Original predictions must use identical physical IDs')
    args.out.mkdir(parents=True)
    started = time.monotonic()
    state = dict(status='RUNNING', commit=args.commit, source=SOURCE, source_commit=SOURCE_COMMIT,
                 previous_ground=GROUND, views={}, TX_truth_reads=0, model_updates=0)
    def save():
        tmp = args.out/'progress.tmp'
        tmp.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n', encoding='utf8')
        os.replace(tmp, args.out/'progress.json')
    save()
    with torch.no_grad():
        for view in core.VIEWS:
            iq = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
            raw = np.load(previous/(view+'.raw.npy'), mmap_mode='r', allow_pickle=False)
            restored = np.load(previous/(view+'.restored.npy'), mmap_mode='r', allow_pickle=False)
            if (iq.shape != (len(ids), 2, 256) or iq.dtype != np.float32 or raw.shape != (len(ids),)
                    or restored.shape != (len(ids),) or raw.dtype != np.int64 or restored.dtype != np.int64):
                raise ValueError('Complete original Phase1 IQ and prediction arrays required')
            counts = dict(processed=0, raw_mismatches=0, restored_mismatches=0)
            state['views'][view] = counts
            for start in range(0, len(ids), 8):
                x = core.tensor(iq[start:start+8], 'cpu')
                result = logits(x, model, dura, reader, include_anchor=False)
                end = start+len(x)
                a, b = compare_batch(result, raw[start:end], restored[start:end])
                counts['processed'] = end
                counts['raw_mismatches'] += a
                counts['restored_mismatches'] += b
                if end % 16384 == 0:
                    state['elapsed_seconds'] = time.monotonic()-started
                    save()
                    print('PAIR_REPLAY_PROGRESS', view, end, 'mismatches', counts['raw_mismatches'],
                          counts['restored_mismatches'], flush=True)
            save()
            print('PAIR_REPLAY_VIEW_COMPLETE', view, counts['processed'], flush=True)
    for obj, snapshot in zip(models, snapshots):
        if any(not torch.equal(value, snapshot[key]) for key, value in obj.state_dict().items()):
            raise ValueError('Frozen model state changed during replay')
    state['elapsed_seconds'] = time.monotonic()-started
    state['status'] = ('PAIR_REPLAY_EXACT' if all(v['raw_mismatches'] == 0 and v['restored_mismatches'] == 0
                      for v in state['views'].values()) else 'PAIR_REPLAY_MISMATCH')
    state['views_complete'] = len(state['views'])
    state['records_per_view'] = len(ids)
    save()
    print(state['status'], flush=True)


if __name__ == '__main__':
    main()
