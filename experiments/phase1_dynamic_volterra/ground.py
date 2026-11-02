"""Independent dynamic Volterra pipeline, fixed source replay and Phase1 ground prediction."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.phase1_dynamic_volterra.model import DynamicVolterra
from experiments.phase1_dynamic_volterra.train_source import load_components, predict_batch


SOURCE = '20261102-phase1-dynamic-volterra-source-s2026092701-r01'
BATCH = 8


def dependencies():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
    import source_pilot as deps
    return deps


def load_fixed(root, commit, device, deps):
    folder = root/'runs'/SOURCE
    done = deps.core.read(folder/'prediction_complete.json')
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit, count=672, fit_ids=8192,
                    F_frozen=True, blind_frozen=True, gate_frozen=True, DURA_loaded=False,
                    TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0)
    if any(done.get(key) != value for key, value in expected.items()):
        raise ValueError('Complete fixed source predictions and frozen components are required')
    state = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
    if (state.get('schema') != 'phase1_dynamic_volterra_v1' or state.get('commit') != commit
            or state.get('epochs') != 5 or len(state.get('fit_ids', [])) != 8192):
        raise ValueError('Exact fixed source-trained dynamic Volterra checkpoint is required')
    model = DynamicVolterra().to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    blind, gate, stats, eye = load_components(root, device, deps)
    reader, _ = deps.core.teacher(root, device)
    return model, blind, gate, stats, eye, reader.eval().requires_grad_(False)


@torch.no_grad()
def smoke(root, out, commit, device, deps):
    if out.exists():
        raise FileExistsError('A new ground output root is required')
    with np.load(root/'runs'/SOURCE/'source_predictions.npz', allow_pickle=False) as z:
        saved = {key: z[key].copy() for key in ('indices', 'rx', 'scene', 'values')}
    if saved['indices'].shape != (672,) or saved['values'].shape != (672, 9):
        raise ValueError('All fixed source predictions are required')
    positions = []
    for group in ((4, 6), (8,)):
        for view in deps.core.VIEWS:
            match = np.flatnonzero(np.isin(saved['rx'], group) & (saved['scene'] == view))
            if len(match) < 8:
                raise ValueError('Eight source records per receiver/scene cell are required')
            positions.extend(match[:8].tolist())
    positions = np.asarray(positions, dtype=np.int64)
    ids = saved['indices'][positions]
    cache, _, rx, _, _, _ = deps.s.source_contract(root)
    received, scene = deps.s.epoch_data(cache, 19)
    if not np.array_equal(rx[ids], saved['rx'][positions]) or not np.array_equal(scene[ids], saved['scene'][positions]):
        raise ValueError('Exact source physical IDs and scenes are required')
    model, blind, gate, stats, eye, reader = load_fixed(root, commit, device, deps)
    rows = []
    for start in range(0, len(ids), BATCH):
        x = deps.core.tensor(received[ids[start:start+BATCH]], device)
        _, _, active, classes = predict_batch(x, model, blind, gate, stats, eye, reader, deps)
        rows.append(torch.stack((active.long(), *classes), 1).cpu())
    actual = np.from_dlpack(torch.cat(rows).contiguous())
    expected = saved['values'][positions][:, [2, 3, 5, 7]].astype(np.int64)
    mismatch = int(np.count_nonzero(np.any(actual != expected, axis=1)))
    if mismatch:
        raise ValueError(f'Fixed source replay differs on {mismatch} records')
    out.mkdir(parents=True)
    deps.core.write(out/'prediction_smoke.json', dict(status='PASS', commit=commit, source_only=True,
                    records=len(ids), mismatches=mismatch, TX_truth_reads=0, U_s_truth_reads=0,
                    V_reads=0, test_IQ_reads=0))
    print('DYNAMIC_VOLTERRA_SOURCE_REPLAY_PASS', len(ids), flush=True)


@torch.no_grad()
def predict(root, out, commit, device, deps):
    check = deps.core.read(out/'prediction_smoke.json')
    if check.get('status') != 'PASS' or check.get('commit') != commit or check.get('test_IQ_reads') != 0:
        raise ValueError('Source-only replay must pass before ground prediction')
    model, blind, gate, stats, eye, reader = load_fixed(root, commit, device, deps)
    folder, ids = deps.core.reference(root)
    reference = root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
    if not np.array_equal(ids, np.load(reference/'physical_ids.npy', allow_pickle=False)):
        raise ValueError('Matching frozen baseline ground IDs are required')
    destination = out/'flow/test_prediction'
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination/'physical_ids.npy', ids, allow_pickle=False)
    accepted = {}
    started = time.monotonic()
    for view in deps.core.VIEWS:
        values = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        original = np.load(reference/(view+'.raw.npy'), mmap_mode='r', allow_pickle=False)
        if values.shape != (len(ids), 2, 256) or values.dtype != np.float32 or original.shape != (len(ids),):
            raise ValueError('Complete fixed Phase1 ground inputs are required')
        raw = np.lib.format.open_memmap(destination/(view+'.raw.npy'), mode='w+', dtype=np.int64, shape=(len(ids),))
        restored = np.lib.format.open_memmap(destination/(view+'.restored.npy'), mode='w+', dtype=np.int64, shape=(len(ids),))
        count = 0
        for start in range(0, len(ids), BATCH):
            x = deps.core.tensor(values[start:start+BATCH], device)
            _, _, active, classes = predict_batch(x, model, blind, gate, stats, eye, reader, deps)
            a = np.from_dlpack(classes[0].cpu().contiguous())
            b = np.from_dlpack(classes[2].cpu().contiguous())
            end = start+len(x)
            if not np.array_equal(a, original[start:end]):
                raise ValueError(f'Frozen raw baseline differs: {view} at {start}')
            raw[start:end], restored[start:end] = a, b
            count += int(active.sum())
            if end % 16384 == 0:
                print('DYNAMIC_VOLTERRA_GROUND_PROGRESS', view, end, round(time.monotonic()-started, 1), flush=True)
        raw.flush()
        restored.flush()
        accepted[view] = count
        print('DYNAMIC_VOLTERRA_GROUND_VIEW_COMPLETE', view, len(ids), flush=True)
    deps.core.write(destination/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids),
                    views=list(deps.core.VIEWS), F_run=deps.core.F_RUN, F_row=deps.core.F_ROW,
                    commit=commit, TX_truth_reads=0, model_updates=0, accepted_by_view=accepted))
    print('DYNAMIC_VOLTERRA_GROUND_PREDICTIONS_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke', 'predict', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deps = dependencies()
    if args.mode == 'smoke':
        smoke(args.root, args.out, args.commit, args.device, deps)
    elif args.mode == 'predict':
        predict(args.root, args.out, args.commit, args.device, deps)
    else:
        if args.device != 'cpu':
            raise ValueError('Independent truth-side scoring must use CPU')
        deps.core.score(args.root, args.out, args.commit)


if __name__ == '__main__':
    main()
