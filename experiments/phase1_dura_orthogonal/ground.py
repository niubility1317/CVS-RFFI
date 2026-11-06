"""Fixed Phase1 ground prediction and independent truth-last scoring."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import SOURCE_PILOT, source_inputs
from experiments.phase1_dura_orthogonal.model import OrthogonalRecovery
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, TRAIN_EPOCHS, FIT_SIZE, extract_context


def load_fixed(root, source, commit):
    folder = root/'runs'/source
    summary = core.read(folder/'source_summary.json')
    expected = dict(status='SOURCE_TRAINED', commit=commit, fit_ids=FIT_SIZE, epochs=TRAIN_EPOCHS,
                    TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                    target_reads=0, DURA_frozen=True)
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('Completed fixed source-only correction required')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    if (state.get('schema') != SCHEMA or state.get('commit') != commit or
            state.get('epochs') != TRAIN_EPOCHS or len(state.get('fit_ids', [])) != FIT_SIZE or
            any(state.get(k) != v for k, v in expected.items() if k not in ('status', 'fit_ids', 'epochs'))):
        raise ValueError('Fixed correction checkpoint and provenance required')
    model = OrthogonalRecovery(width=state['width'])
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    dura = frozen_dura(root, 'cpu')
    reader, _ = core.teacher(root, 'cpu')
    return model, dura, reader.eval().requires_grad_(False)


@torch.no_grad()
def decisions(x, model, dura, reader):
    anchor, direction, variance = extract_context(dura, x)
    restored = model(x, anchor, direction, variance)
    outputs = []
    for signal in (x, anchor, restored):
        logits = reader.classify_features(reader.features(signal))
        if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
            raise ValueError('Finite frozen reader outputs required')
        outputs.append(logits.argmax(1))
    return outputs


def smoke(root, out, source, commit):
    if out.exists():
        raise FileExistsError('A new ground output root is required')
    ids, rx, scene, _, received, _ = source_inputs(root)
    positions = []
    for receivers in ((4, 6), (8,)):
        for view in core.VIEWS:
            match = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(match) < 8:
                raise ValueError('Eight source observations per receiver group and view required')
            positions.extend(match[:8].tolist())
    positions = np.asarray(positions, dtype=np.int64)
    with np.load(root/'runs'/SOURCE_PILOT/'source_predictions.npz', allow_pickle=False) as z:
        expected = z['values'][positions][:, [3, 5]].astype(np.int64)
    model, dura, reader = load_fixed(root, source, commit)
    actual = []
    for start in range(0, len(positions), 8):
        x = core.tensor(received[ids[positions[start:start+8]]], 'cpu')
        raw, anchor, restored = decisions(x, model, dura, reader)
        actual.append(torch.stack((raw, anchor), 1).cpu())
        if not torch.isfinite(restored).all():
            raise ValueError('Finite source-only restored decisions required')
    actual = np.from_dlpack(torch.cat(actual).contiguous()).copy()
    if not np.array_equal(actual, expected):
        raise ValueError('Frozen raw and DURA source decisions differ from fixed baseline')
    out.mkdir(parents=True)
    core.write(out/'prediction_smoke.json', dict(status='PASS', commit=commit, source_only=True,
        records=len(positions), raw_and_dura_mismatches=0, TX_truth_reads=0,
        U_s_truth_reads=0, V_reads=0, test_IQ_reads=0))
    print('ORTHOGONAL_SOURCE_PREDICTION_SMOKE_PASS', len(positions), flush=True)


def predict(root, out, source, commit):
    check = core.read(out/'prediction_smoke.json')
    if check.get('status') != 'PASS' or check.get('commit') != commit or check.get('test_IQ_reads') != 0:
        raise ValueError('Matching source-only prediction smoke required')
    model, dura, reader = load_fixed(root, source, commit)
    folder, ids = core.reference(root)
    reference = root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
    if not np.array_equal(ids, np.load(reference/'physical_ids.npy', allow_pickle=False)):
        raise ValueError('Matching fixed baseline physical IDs required')
    destination = out/'flow/test_prediction'
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination/'physical_ids.npy', ids, allow_pickle=False)
    started = time.monotonic()
    for view in core.VIEWS:
        values = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        original = np.load(reference/(view+'.raw.npy'), mmap_mode='r', allow_pickle=False)
        if values.shape != (len(ids), 2, 256) or values.dtype != np.float32 or original.shape != (len(ids),):
            raise ValueError('Complete fixed Phase1 inputs required')
        raw = np.lib.format.open_memmap(destination/(view+'.raw.npy'), mode='w+', dtype=np.int64, shape=(len(ids),))
        restored = np.lib.format.open_memmap(destination/(view+'.restored.npy'), mode='w+', dtype=np.int64, shape=(len(ids),))
        for start in range(0, len(ids), 8):
            x = core.tensor(values[start:start+8], 'cpu')
            a, _, b = decisions(x, model, dura, reader)
            a = np.from_dlpack(a.cpu().contiguous())
            b = np.from_dlpack(b.cpu().contiguous())
            end = start+len(x)
            if not np.array_equal(a, original[start:end]):
                raise ValueError('Frozen raw baseline differs at '+view+' '+str(start))
            raw[start:end], restored[start:end] = a, b
            if end % 16384 == 0:
                print('ORTHOGONAL_GROUND_PROGRESS', view, end, round(time.monotonic()-started, 1), flush=True)
        raw.flush()
        restored.flush()
        print('ORTHOGONAL_GROUND_VIEW_COMPLETE', view, len(ids), flush=True)
    core.write(destination/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids),
        views=list(core.VIEWS), F_run=core.F_RUN, F_row=core.F_ROW, commit=commit,
        TX_truth_reads=0, model_updates=0))
    print('ORTHOGONAL_GROUND_PREDICTIONS_COMPLETE', flush=True)


def paired_score(root, out, commit):
    """Report DURA-paired accuracy after all candidate predictions are fixed."""
    candidate = out/'flow/test_prediction'
    reference = root/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction'
    done = core.read(candidate/'complete.json')
    core.validate_prediction_manifest(done, count=168000, commit=commit)
    ids = np.load(candidate/'physical_ids.npy', allow_pickle=False)
    if ids.shape != (168000,) or not np.array_equal(ids, np.load(reference/'physical_ids.npy', allow_pickle=False)):
        raise ValueError('The DURA pair must use identical physical IDs')
    predictions = {}
    for view in core.VIEWS:
        raw = np.load(candidate/(view+'.raw.npy'), allow_pickle=False)
        restored = np.load(candidate/(view+'.restored.npy'), allow_pickle=False)
        reference_raw = np.load(reference/(view+'.raw.npy'), allow_pickle=False)
        reference_restored = np.load(reference/(view+'.restored.npy'), allow_pickle=False)
        if (any(z.shape != ids.shape or z.dtype != np.int64 for z in (raw, restored, reference_raw, reference_restored))
                or not np.array_equal(raw, reference_raw)
                or any(((z < 0) | (z >= 6)).any() for z in (raw, restored, reference_restored))):
            raise ValueError('Complete matched frozen decision arrays required')
        predictions[view] = (restored, reference_restored)
    truth = core.read(root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    labels = np.asarray([truth[str(i)]['label'] for i in ids])
    views = {}
    for view, (candidate_labels, baseline_labels) in predictions.items():
        candidate_correct = candidate_labels == labels
        baseline_correct = baseline_labels == labels
        views[view] = dict(count=len(ids), candidate_correct=int(candidate_correct.sum()),
                           baseline_correct=int(baseline_correct.sum()),
                           repairs=int((candidate_correct & ~baseline_correct).sum()),
                           damages=int((~candidate_correct & baseline_correct).sum()))
    core.write(out/'flow/paired_dura.json', dict(status='PAIRED_COMPLETE', count=len(ids), views=views))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke', 'predict', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode == 'smoke':
        smoke(args.root, args.out, args.source, args.commit)
    elif args.mode == 'predict':
        predict(args.root, args.out, args.source, args.commit)
    else:
        core.score(args.root, args.out, args.commit)
        paired_score(args.root, args.out, args.commit)


if __name__ == '__main__':
    main()
