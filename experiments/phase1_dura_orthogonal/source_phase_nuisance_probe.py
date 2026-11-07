"""Compare fixed versus rotated phase targets on source-only DURA outputs."""
import argparse
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle
from experiments.phase1_dura_orthogonal.source_phase_orbit_probe import rotate


def evaluate_batch(dura, received, clean):
    import torch

    if received.shape != clean.shape or received.ndim != 3 or received.shape[1:] != (2, 256):
        raise ValueError('Matching source IQ windows required')
    with torch.no_grad():
        baseline = dura(received)
        energy = clean.square().mean((1, 2)).clamp_min(1e-8)
        rows = []
        for k in range(4):
            out = baseline if k == 0 else dura(rotate(received, k))
            refs = (clean, rotate(clean, k), baseline, rotate(baseline, k))
            rows.append(torch.stack([(out - ref).square().mean((1, 2)) / energy for ref in refs], 1))
        values = torch.stack(rows, 1)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Finite source waveform metrics required')
        return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    uuid = wait_for_idle(args.out, args.gpu, allow_shared=True)
    if 'torch' in sys.modules:
        raise RuntimeError('GPU must be selected before importing torch')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs

    if not torch.cuda.is_available():
        raise ValueError('Registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    dura = frozen_dura(args.root, 'cuda:0')
    def inputs(chosen):
        return core.tensor(received[chosen], 'cuda:0'), core.tensor(clean[chosen], 'cuda:0')
    smoke = evaluate_batch(dura, *inputs(ids[:8]))
    if smoke.shape != (8, 4, 4) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen real-checkpoint source smoke required')
    print('SOURCE_PHASE_NUISANCE_SMOKE_PASS', len(smoke), flush=True)
    all_values = torch.cat([evaluate_batch(dura, *inputs(ids[i:i+16])).cpu()
                            for i in range(0, len(ids), 16)])
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = torch.as_tensor((np.isin(rx, receivers) & (scene == view)).tolist(), dtype=torch.bool)
            cell = all_values[mask]
            if len(cell) != 48:
                raise ValueError('48 source records per receiver/view group required')
            groups[name+'/'+view] = dict(count=len(cell), mean_metrics=cell.mean(0).tolist(),
                canonical_closer_counts=(cell[:, :, 0] < cell[:, :, 1]).sum(0).tolist())
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_PHASE_NUISANCE_COMPLETE', commit=args.commit,
        records=len(ids), quarter_turns=[0, 1, 2, 3], groups=groups,
        metric_order=['unrotated_clean_nmse', 'rotated_clean_nmse',
                      'invariance_disagreement_nmse', 'equivariance_disagreement_nmse'],
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_reads=0, model_updates=0, physical_gpu=args.gpu, physical_gpu_uuid=uuid,
        caveat='Synthetic input rotations on explored source IQ; no claim about the physical channel distribution or identity accuracy.'))
    print('SOURCE_PHASE_NUISANCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
