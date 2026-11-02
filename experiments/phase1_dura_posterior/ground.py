"""Fixed source-trained DURA posterior model and Phase1 truth-last evaluation."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_posterior.integrated import PosteriorInDura
from experiments.phase1_dura_posterior.physical_source_smoke import load_source_models, s
from experiments.phase1_dura_posterior.pipeline import GatedPosteriorDura


SOURCE = '20261102-phase1-dura-posterior-internal-source-s2026092701-r01'
NONLOCAL_SOURCE = '20261102-phase1-dura-nonlocal-internal-source-s2026092701-r01'
BATCH = 8


def load_fixed(root, commit, device, source_run):
    source = root/'runs'/source_run
    metadata = core.read(source/'prediction_complete.json')
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit, count=672, fit_ids=8192,
                    F_frozen=True, DURA_frozen=True, gate_frozen=True, posterior_frozen=True,
                    TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0,
                    model_updates_after_prediction=0)
    kind = 'mean' if source_run == SOURCE else 'nonlocal'
    expected['posterior_kind'] = kind
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('The fixed source-trained model and complete source predictions are required')
    state = torch.load(source/'checkpoint.pt', map_location='cpu', weights_only=False)
    if (state.get('schema') != 'phase1_dura_posterior_source_v1' or state.get('commit') != commit
            or state.get('epochs') != 5 or state.get('posterior_kind') != kind
            or len(state.get('fit_ids', [])) != 8192):
        raise ValueError('Exact source checkpoint required')
    reader, _ = core.teacher(root, device)
    dura = frozen_dura(root, device)
    integrated = PosteriorInDura(dura).to(device)
    integrated.corrections.load_state_dict(state['corrections'], strict=True)
    physical, gate = load_source_models(root, kind)
    pipe = GatedPosteriorDura(integrated, gate.to(device), physical.to(device)).to(device)
    return pipe.eval().requires_grad_(False), reader.eval().requires_grad_(False)


@torch.no_grad()
def source_smoke(root, out, commit, device, source_run):
    if out.exists():
        raise FileExistsError('A unique ground run is required')
    source = root/'runs'/source_run
    with np.load(source/'source_predictions.npz', allow_pickle=False) as z:
        saved = {name: z[name].copy() for name in ('indices', 'rx', 'scene', 'values')}
    if saved['indices'].shape != (672,) or saved['values'].shape != (672, 9):
        raise ValueError('All fixed source predictions are required')
    positions = []
    for group in ((4, 6), (8,)):
        for view in core.VIEWS:
            match = np.flatnonzero(np.isin(saved['rx'], group) & (saved['scene'] == view))
            if len(match) < 8:
                raise ValueError('Eight fixed source records per receiver/scene are required')
            positions.extend(match[:8].tolist())
    positions = np.asarray(positions, dtype=np.int64)
    indices = saved['indices'][positions]
    cache, _, rx, _, _, _ = s.source_contract(root)
    received, scene = s.epoch_data(cache, 19)
    if not np.array_equal(rx[indices], saved['rx'][positions]) or not np.array_equal(scene[indices], saved['scene'][positions]):
        raise ValueError('Source physical records and scenes must match')
    pipe, reader = load_fixed(root, commit, device, source_run)
    rows = []
    for start in range(0, len(indices), BATCH):
        x = core.tensor(received[indices[start:start+BATCH]], device)
        dura = pipe.integrated.dura(x)
        output, active = pipe(x)
        classes = [reader.classify_features(reader.features(v)).argmax(1) for v in (x, dura, output)]
        rows.append(torch.stack((active.long(), *classes), 1).cpu())
    actual = np.from_dlpack(torch.cat(rows).contiguous())
    expected = saved['values'][positions][:, [2, 3, 5, 7]].astype(np.int64)
    mismatches = int(np.count_nonzero(np.any(actual != expected, axis=1)))
    if mismatches:
        raise ValueError(f'Fixed source replay differs for {mismatches} records')
    out.mkdir(parents=True)
    core.write(out/'prediction_smoke.json', dict(status='PASS', commit=commit, source_only=True,
               source_run=source_run, records=len(indices), mismatches=mismatches, TX_truth_reads=0, U_s_truth_reads=0,
               V_reads=0, test_IQ_reads=0))
    print('DURA_POSTERIOR_SOURCE_REPLAY_PASS', len(indices), flush=True)


@torch.no_grad()
def predict(root, out, commit, device, source_run):
    check = core.read(out/'prediction_smoke.json')
    if (check.get('status') != 'PASS' or check.get('commit') != commit
            or check.get('source_run') != source_run or check.get('test_IQ_reads') != 0):
        raise ValueError('Source-only replay must pass before ground prediction')
    pipe, reader = load_fixed(root, commit, device, source_run)
    folder, ids = core.reference(root)
    reference = root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
    if not np.array_equal(ids, np.load(reference/'physical_ids.npy', allow_pickle=False)):
        raise ValueError('The exact frozen DURA comparison IDs are required')
    destination = out/'flow/test_prediction'
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination/'physical_ids.npy', ids, allow_pickle=False)
    accepted = {}
    started = time.monotonic()
    for view in core.VIEWS:
        values = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        raw_reference = np.load(reference/(view+'.raw.npy'), mmap_mode='r', allow_pickle=False)
        if values.shape != (len(ids), 2, 256) or values.dtype != np.float32 or raw_reference.shape != (len(ids),):
            raise ValueError('Complete fixed Phase1 ground arrays are required')
        raw = np.lib.format.open_memmap(destination/(view+'.raw.npy'), mode='w+', dtype=np.int64, shape=(len(ids),))
        restored = np.lib.format.open_memmap(destination/(view+'.restored.npy'), mode='w+', dtype=np.int64, shape=(len(ids),))
        active_count = 0
        for start in range(0, len(ids), BATCH):
            x = core.tensor(values[start:start+BATCH], device)
            output, active = pipe(x)
            a = reader.classify_features(reader.features(x)).argmax(1).cpu().contiguous()
            b = reader.classify_features(reader.features(output)).argmax(1).cpu().contiguous()
            a = np.from_dlpack(a)
            b = np.from_dlpack(b)
            end = start+len(x)
            if not np.array_equal(a, raw_reference[start:end]):
                raise ValueError(f'Fixed raw baseline differs: {view} at {start}')
            raw[start:end], restored[start:end] = a, b
            active_count += int(active.sum())
            if end % 16384 == 0:
                print('DURA_POSTERIOR_GROUND_PROGRESS', view, end, round(time.monotonic()-started, 1), flush=True)
        raw.flush()
        restored.flush()
        accepted[view] = active_count
        print('DURA_POSTERIOR_GROUND_VIEW_COMPLETE', view, len(ids), flush=True)
    core.write(destination/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids), views=list(core.VIEWS),
               F_run=core.F_RUN, F_row=core.F_ROW, commit=commit, TX_truth_reads=0, model_updates=0,
               accepted_by_view=accepted))
    print('DURA_POSTERIOR_GROUND_PREDICTIONS_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke', 'predict', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--source-run', choices=(SOURCE, NONLOCAL_SOURCE), default=SOURCE)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == 'smoke':
        source_smoke(args.root, args.out, args.commit, args.device, args.source_run)
    elif args.mode == 'predict':
        predict(args.root, args.out, args.commit, args.device, args.source_run)
    else:
        if args.device != 'cpu':
            raise ValueError('Independent truth scoring must use CPU')
        core.score(args.root, args.out, args.commit)


if __name__ == '__main__':
    main()
