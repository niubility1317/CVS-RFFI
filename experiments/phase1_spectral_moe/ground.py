"""Fixed Phase1 seven-view prediction and independent truth-last scoring."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import SOURCE_PILOT, source_inputs
from experiments.phase1_dura_orthogonal.ground import paired_score
from experiments.phase1_joint_particle.particle import JointParticleRecovery, PARTICLES, STEPS
from experiments.phase1_joint_particle.train import SCHEMA, EPOCHS, WIDTH


from experiments.phase1_spectral_moe.model import SpectralMixture, CombinedRecovery
from experiments.phase1_spectral_moe.train import SCHEMA as MIX_SCHEMA, EPOCHS as MIX_EPOCHS, WIDTH as MIX_WIDTH
from experiments.phase1_residual_diffusion.source_probe import load_model, JOINT, JOINT_COMMIT, DIFFUSION, DIFFUSION_COMMIT
from experiments.phase1_residual_diffusion.model import ResidualDiffusion
from experiments.phase1_residual_diffusion.train import SCHEMA as DIFFUSION_SCHEMA
from experiments.phase1_joint_particle.train import SCHEMA as JOINT_SCHEMA

def load_fixed(root, source, commit, device, *, preflight=False):
    folder = root/'runs'/(source+'-training-smoke' if preflight else source)
    summary = core.read(folder/('smoke.json' if preflight else 'source_summary.json'))
    state = torch.load(folder/('smoke.pt' if preflight else 'final.pt'), map_location='cpu', weights_only=True)
    expected = dict(schema=MIX_SCHEMA, commit=commit, epochs=1 if preflight else MIX_EPOCHS,
        fit_ids=17632, actual_fit_ids=64 if preflight else 17632,
        actual_source_view_count=192 if preflight else 52896, views_per_record=3, width=MIX_WIDTH,
        joint=JOINT, joint_commit=JOINT_COMMIT, diffusion=DIFFUSION, diffusion_commit=DIFFUSION_COMMIT,
        parent_fit_disjoint=True, pilot_disjoint=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if summary.get('status') != ('SOURCE_TRAINING_SMOKE_PASS' if preflight else 'SOURCE_TRAINED') or any(
            obj.get(key) != value for obj in (summary, state) for key, value in expected.items()):
        raise ValueError('Fixed source-trained mixture and matching provenance required')
    gate = SpectralMixture(MIX_WIDTH).to(device)
    gate.load_state_dict(state['model'], strict=True)
    joint = load_model(root, JOINT, JOINT_SCHEMA, JOINT_COMMIT, 3, JointParticleRecovery)
    diffusion = load_model(root, DIFFUSION, DIFFUSION_SCHEMA, DIFFUSION_COMMIT, 8, ResidualDiffusion)
    model = CombinedRecovery(gate, joint, diffusion).eval().requires_grad_(False)
    dura = frozen_dura(root, device)
    reader, _ = core.teacher(root, 'cpu')
    return model, dura, reader.eval().requires_grad_(False)


@torch.no_grad()
def decisions(received_cpu, model, dura, reader, device):
    """Frozen reader remains on CPU; both restorers consume the same IQ."""
    raw = reader.classify_features(reader.features(received_cpu))
    received = received_cpu.to(device)
    context = dura.estimate_context(received)
    anchor = dura.stage_outputs(received, context)[0][-1]
    action, variance = context[1], context[2]
    observed = torch.complex(received[:, 0], received[:, 1])
    initial = torch.complex(anchor[:, 0], anchor[:, 1])
    residual = action.adjoint(observed - action(initial))
    innovation = torch.stack((residual.real, residual.imag), dim=1)
    restored = model(received, anchor, innovation, variance, action)
    logits = [raw]
    for signal in (anchor, restored):
        current = reader.classify_features(reader.features(signal.to('cpu')))
        logits.append(current)
    if any(x.shape != (len(received_cpu), 6) or not torch.isfinite(x).all() for x in logits):
        raise ValueError('Finite six-class frozen reader outputs required')
    return tuple(x.argmax(1) for x in logits)


def smoke(root, out, source, commit, device):
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
    model, dura, reader = load_fixed(root, source, commit, device, preflight=True)
    actual = []
    for start in range(0, len(positions), 8):
        x = core.tensor(received[ids[positions[start:start+8]]], 'cpu')
        raw, baseline, _ = decisions(x, model, dura, reader, device)
        actual.append(torch.stack((raw, baseline), 1).cpu())
    actual = np.from_dlpack(torch.cat(actual).contiguous()).copy()
    if not np.array_equal(actual, expected):
        raise ValueError('Frozen raw and DURA source decisions differ from fixed baseline')
    out.mkdir(parents=True)
    core.write(out/'prediction_smoke.json', dict(status='PASS', commit=commit, source_only=True,
        records=len(positions), raw_and_dura_mismatches=0, TX_truth_reads=0,
        U_s_truth_reads=0, V_reads=0, test_IQ_reads=0, source=source, device=device))
    print('SPECTRAL_SOURCE_PREDICTION_SMOKE_PASS', len(positions), flush=True)


def predict(root, out, source, commit, device):
    if out.exists():
        raise FileExistsError('A new ground output root is required')
    model, dura, reader = load_fixed(root, source, commit, device)
    out.mkdir(parents=True)
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
            a, _, b = decisions(x, model, dura, reader, device)
            a, b = np.from_dlpack(a.contiguous()), np.from_dlpack(b.contiguous())
            end = start + len(x)
            if not np.array_equal(a, original[start:end]):
                raise ValueError('Frozen raw baseline differs at '+view+' '+str(start))
            raw[start:end], restored[start:end] = a, b
            if end % 16384 == 0:
                print('SPECTRAL_GROUND_PROGRESS', view, end, round(time.monotonic()-started, 1), flush=True)
        raw.flush()
        restored.flush()
        print('SPECTRAL_GROUND_VIEW_COMPLETE', view, len(ids), flush=True)
    core.write(out/'flow/inference.json', dict(source=source, source_commit=commit,
        inference_mode='single_IQ_three_expert_frequency_mixture', device=device, commit=commit))
    core.write(destination/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids),
        views=list(core.VIEWS), F_run=core.F_RUN, F_row=core.F_ROW, commit=commit,
        TX_truth_reads=0, model_updates=0))
    print('SPECTRAL_GROUND_PREDICTIONS_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke', 'predict', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.mode != 'score':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    if args.mode != 'score' and (args.device != 'cuda:0' or not torch.cuda.is_available()):
        raise ValueError('One explicitly selected CUDA GPU required for prediction')
    if args.mode == 'smoke':
        smoke(args.root, args.out, args.source, args.commit, args.device)
    elif args.mode == 'predict':
        predict(args.root, args.out, args.source, args.commit, args.device)
    else:
        core.score(args.root, args.out, args.commit)
        paired_score(args.root, args.out, args.commit)


if __name__ == '__main__':
    main()
