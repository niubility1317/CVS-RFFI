"""Source-only paired IQ comparison of two fixed recovery checkpoints."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.train_source import load_source_training, WIDE_FIT_SIZE
from experiments.phase1_joint_particle.particle import JointParticleRecovery
from experiments.phase1_joint_particle.train import SCHEMA as JOINT_SCHEMA
from experiments.phase1_residual_diffusion.model import ResidualDiffusion
from experiments.phase1_residual_diffusion.train import SCHEMA as DIFFUSION_SCHEMA

JOINT = '20261108-phase1-dura-joint-particle-source-s2026092701-r01'
JOINT_COMMIT = '0cddb5ff260381205a5445d04507ed211fac9989'
DIFFUSION = '20261108-phase1-dura-residual-diffusion-source-s2026092701-r01'
DIFFUSION_COMMIT = 'c48f5bebc921b4e2fd97341cb8eebcf6a627dbe2'


def check_metadata(summary, state, *, schema, commit, steps):
    required = dict(schema=schema, commit=commit, epochs=20, fit_ids=32768,
        actual_source_view_count=98304, views_per_record=3, width=192, steps=steps,
        inference_particles=4, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, DURA_frozen=True)
    if summary.get('status') != 'SOURCE_TRAINED' or any(
            obj.get(k) != v for obj in (summary, state) for k, v in required.items()):
        raise ValueError('Fixed source-only checkpoint and metadata required')


def load_model(root, name, schema, commit, steps, factory):
    folder = root/'runs'/name
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=True)
    check_metadata(summary, state, schema=schema, commit=commit, steps=steps)
    model = factory(width=192).to('cuda:0')
    model.load_state_dict(state['model'], strict=True)
    return model.eval().requires_grad_(False)


def normalized_error(signal, clean):
    if signal.shape != clean.shape or signal.ndim != 3 or signal.shape[1:] != (2, 256):
        raise ValueError('Matched 256-sample IQ required')
    if not torch.isfinite(signal).all() or not torch.isfinite(clean).all():
        raise ValueError('Finite source IQ required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-12)
    return (signal-clean).square().mean((1, 2))/energy


@torch.no_grad()
def evaluate(dura, joint, diffusion, received, clean):
    context = dura.estimate_context(received)
    anchor = dura.stage_outputs(received, context)[0][-1]
    action, variance = context[1], context[2]
    y = torch.complex(received[:, 0], received[:, 1])
    z = torch.complex(anchor[:, 0], anchor[:, 1])
    residual = action.adjoint(y-action(z))
    innovation = torch.stack((residual.real, residual.imag), 1)
    outputs = {'dura': anchor,
        'joint': joint(received, anchor, innovation, variance, action),
        'diffusion': diffusion(received, anchor, innovation, variance)}
    return {name: normalized_error(signal, clean) for name, signal in outputs.items()}


def run(root, out, commit):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if out.exists():
        raise FileExistsError('A new source diagnostic output root is required')
    if not torch.cuda.is_available():
        raise ValueError('An explicitly selected GPU is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, rx, scene, _, received, clean = source_inputs(root)
    _, _, _, fit = load_source_training(root, ids, fit_size=WIDE_FIT_SIZE)
    if len(fit) != 32768 or np.intersect1d(ids, fit).size:
        raise ValueError('Source diagnostic records must be excluded from candidate fits')
    dura = frozen_dura(root, 'cuda:0')
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Original DURA must be frozen')
    joint = load_model(root, JOINT, JOINT_SCHEMA, JOINT_COMMIT, 3, JointParticleRecovery)
    diffusion = load_model(root, DIFFUSION, DIFFUSION_SCHEMA, DIFFUSION_COMMIT, 8, ResidualDiffusion)
    evaluate(dura, joint, diffusion, core.tensor(received[ids[:2]], 'cuda:0'), core.tensor(clean[ids[:2]], 'cuda:0'))
    print('RECOVERY_PILOT_SMOKE_PASS 2', flush=True)
    out.mkdir(parents=True)
    chunks = {'dura': [], 'joint': [], 'diffusion': []}
    started = time.monotonic()
    for start in range(0, len(ids), 8):
        selected = ids[start:start+8]
        values = evaluate(dura, joint, diffusion, core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0'))
        for name, tensor in values.items():
            chunks[name].append(tensor.detach().cpu().to(torch.float64))
        if (start+len(selected)) % 168 == 0:
            print('RECOVERY_PILOT_PROGRESS', start+len(selected), flush=True)
    arrays = {name: np.from_dlpack(torch.cat(chunks[name]).contiguous()).copy() for name in chunks}
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if int(mask.sum()) != 48:
                raise ValueError('Each fixed source receiver and view requires 48 records')
            reference = arrays['dura'][mask]
            metrics = {'dura_nmse': float(reference.mean())}
            for candidate in ('joint', 'diffusion'):
                values = arrays[candidate][mask]
                metrics[candidate+'_nmse'] = float(values.mean())
                metrics[candidate+'_lower'] = int(np.sum(values < reference-1e-9))
                metrics[candidate+'_higher'] = int(np.sum(values > reference+1e-9))
            groups[name+'/'+view] = dict(count=48, **metrics)
    core.write(out/'result.json', dict(status='SOURCE_IQ_DIAGNOSTIC_COMPLETE', commit=commit,
        joint_commit=JOINT_COMMIT, diffusion_commit=DIFFUSION_COMMIT, records=len(ids), groups=groups,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0, source_only=True,
        caveat='Previously explored source pilot; IQ NMSE is not transmitter identity accuracy.'))
    print('RECOVERY_PILOT_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    run(args.root, args.out, args.commit)


if __name__ == '__main__':
    main()
