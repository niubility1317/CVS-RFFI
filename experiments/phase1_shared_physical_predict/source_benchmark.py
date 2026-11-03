"""Source-only equivalence and timing of fixed shared physical predictions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

# Import all source model code before reader activation can alter package paths.
from experiments.phase1_dura_physical_fusion.model import PhysicalFusion
from experiments.phase1_dura_physical_wide.model import WidePhysicalFusion
from experiments.phase1_dura_physical_fusion.pipeline import load_components, predict_batch
from experiments.phase1_shared_physical_predict.shared import predict_shared

SOURCES = (
    ('20261103-phase1-dura-physical-fusion-source-s2026092701-r02',
     'f54a646afd873f917f16f887e38c0c560fe06a8f', 'phase1_dura_physical_fusion_v1', PhysicalFusion),
    ('20261103-phase1-dura-physical-wide-source-s2026092701-r01',
     '117c5c2ea32013c65391dce5b807cafe2361e33d', 'phase1_dura_physical_wide_v1', WidePhysicalFusion),
    ('20261103-phase1-dura-physical-local-teacher-source-s2026092701-r01',
     '4786a605182976e7da8e26802710fdbca509e8cc', 'phase1_dura_physical_local_teacher_v1', WidePhysicalFusion),
)


def tensor_to_reference_array(tensor):
    """Use this module's NumPy implementation instead of Torch's NumPy bridge."""
    return np.from_dlpack(tensor.detach().cpu().contiguous()).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('A new source-only benchmark output is required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'phase1_joint_mixture'))
    import source_pilot as deps

    pilots, models = [], []
    for name, commit, schema, factory in SOURCES:
        folder = args.root / 'runs' / name
        metadata = json.loads((folder / 'prediction_complete.json').read_text(encoding='utf8'))
        expected = dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit, count=672,
                        fit_ids=8192, F_frozen=True, blind_frozen=True, gate_frozen=True,
                        DURA_loaded=True, DURA_frozen=True, TX_labels_read=0,
                        U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0)
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('All three frozen source candidates are required')
        checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if checkpoint.get('schema') != schema or checkpoint.get('commit') != commit or checkpoint.get('epochs') != 5:
            raise ValueError('Registered checkpoint architecture and provenance required')
        model = factory()
        model.load_state_dict(checkpoint['model'], strict=True)
        models.append(model.eval().requires_grad_(False))
        with np.load(folder / 'source_predictions.npz', allow_pickle=False) as z:
            pilots.append({k: z[k].copy() for k in ('indices', 'rx', 'scene', 'values')})
    for pilot in pilots[1:]:
        for k in ('indices', 'rx', 'scene'):
            if not np.array_equal(pilot[k], pilots[0][k]):
                raise ValueError('All three frozen source pilots must use identical records')
    positions = []
    for receivers in ((4, 6), (8,)):
        for view in deps.core.VIEWS:
            match = np.flatnonzero(np.isin(pilots[0]['rx'], receivers) & (pilots[0]['scene'] == view))
            if len(match) < 8:
                raise ValueError('Eight source records per receiver/scene cell required')
            positions.extend(match[:8].tolist())
    positions = np.asarray(positions, dtype=np.int64)
    ids = pilots[0]['indices'][positions]
    cache, _, rx, _, _, _ = deps.s.source_contract(args.root)
    received, scenes = deps.s.epoch_data(cache, 19)
    if not np.array_equal(rx[ids], pilots[0]['rx'][positions]) or not np.array_equal(scenes[ids], pilots[0]['scene'][positions]):
        raise ValueError('Source records must match the fixed pilot')
    blind, gate, dura, physical = load_components(args.root, 'cpu', deps)
    reader, _ = deps.core.teacher(args.root, 'cpu')
    reader.eval().requires_grad_(False)
    baseline_seconds = 0.
    shared_seconds = 0.
    timed_records = 0
    with torch.no_grad():
        for start in range(0, len(ids), 8):
            x = deps.core.tensor(received[ids[start:start+8]], 'cpu')
            # Alternate execution order to reduce systematic warm-cache bias.
            if (start // 8) % 2 == 0:
                before = time.perf_counter()
                original = [predict_batch(x, model, blind, gate, dura, physical, reader, deps) for model in models]
                normal_elapsed = time.perf_counter() - before
                before = time.perf_counter()
                optimized = predict_shared(x, models, gate, dura, physical, reader)
                shared_elapsed = time.perf_counter() - before
            else:
                before = time.perf_counter()
                optimized = predict_shared(x, models, gate, dura, physical, reader)
                shared_elapsed = time.perf_counter() - before
                before = time.perf_counter()
                original = [predict_batch(x, model, blind, gate, dura, physical, reader, deps) for model in models]
                normal_elapsed = time.perf_counter() - before
            for column, (a, b) in enumerate(zip(original, optimized)):
                if not all(torch.equal(left, right) for left, right in zip(a[:3], b[:3])):
                    raise ValueError('Original and shared waveforms/gates differ')
                if not all(torch.equal(left, right) for left, right in zip(a[3], b[3])):
                    raise ValueError('Original and shared classifications differ')
                actual = tensor_to_reference_array(torch.stack((b[2].long(), *b[3]), 1))
                expected = pilots[column]['values'][positions[start:start+len(x)]][:, [2, 3, 5, 7]].astype(np.int64)
                if not np.array_equal(actual, expected):
                    raise ValueError('Source decisions must reproduce each fixed pilot')
            if start:
                baseline_seconds += normal_elapsed
                shared_seconds += shared_elapsed
                timed_records += len(x)
    if timed_records != 104 or shared_seconds <= 0:
        raise ValueError('Complete timing and source replay required')
    result = dict(status='PASS', source_only=True, records=len(ids), timed_records=timed_records,
                  batch_size=8, cpu_threads=2, device='cpu', numerical_mode='FP32 TF32 disabled',
                  waveform_and_decision_equality=True, fixed_source_pilot_replayed=112,
                  baseline_seconds=baseline_seconds, shared_seconds=shared_seconds,
                  measured_speedup=baseline_seconds/shared_seconds,
                  load_average=list(os.getloadavg()), TX_truth_reads=0, U_s_truth_reads=0, V_reads=0,
                  target_IQ_reads=0, model_updates=0,
                  caveat='Source-only sample and contemporaneous CPU load; no target-output equivalence asserted.')
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
