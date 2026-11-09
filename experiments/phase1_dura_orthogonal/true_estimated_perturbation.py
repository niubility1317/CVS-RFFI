"""Compare source simulator and DURA-estimated responses to identical perturbations."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import DIRECTIONS, EPSILON, perturbations


def action_metrics(estimate, truth):
    """Per-record squared relative error, real alignment, and norm ratio."""
    if (estimate.ndim != 2 or estimate.shape != truth.shape or
            not torch.is_complex(estimate) or not torch.is_complex(truth)):
        raise ValueError('Matching complex action responses required')
    if not torch.isfinite(estimate).all() or not torch.isfinite(truth).all():
        raise FloatingPointError('Finite action responses required')
    true_energy = truth.abs().square().sum(-1).clamp_min(1e-20)
    estimated_energy = estimate.abs().square().sum(-1).clamp_min(1e-20)
    error = (estimate - truth).abs().square().sum(-1) / true_energy
    alignment = (truth.conj() * estimate).real.sum(-1) / (true_energy * estimated_energy).sqrt()
    return torch.stack((error, alignment, (estimated_energy / true_energy).sqrt()), dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or not torch.cuda.is_available():
        raise ValueError('Exactly one explicitly selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    folder = args.root / 'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/epochs/19'
    metadata = core.read(folder / 'complete.json')
    if (metadata.get('count') != core.COUNT or metadata.get('namespace') != 'learned-action/train/epoch19'
            or any(metadata.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
        raise ValueError('Matching legal source simulator action cache required')
    kernel = np.load(folder / 'kernel.npy', mmap_mode='r', allow_pickle=False)
    coefficients = np.load(folder / 'coefficients.npy', mmap_mode='r', allow_pickle=False)
    if (kernel.shape != (core.COUNT, 2, 64) or coefficients.shape != (core.COUNT, 4, 256)
            or kernel.dtype != np.float32 or coefficients.dtype != np.float32):
        raise ValueError('Complete matching source simulator actions required')
    dura = frozen_dura(args.root, 'cuda:0')
    started = time.monotonic()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start + 32]
            x = core.tensor(received[selected], 'cuda:0')
            y = core.tensor(clean[selected], 'cuda:0')
            h = core.tensor(kernel[selected], 'cuda:0')
            c = core.tensor(coefficients[selected], 'cuda:0')
            context = dura.estimate_context(x)
            estimated = context[1]
            simulated = true_action(type(estimated), h, c)
            if start == 0:
                torch.testing.assert_close(dura.stage_outputs(x, context)[0][-1], dura(x), rtol=1e-5, atol=3e-6)
                args.out.mkdir(parents=True)
                print('TRUE_ESTIMATED_ACTION_SOURCE_SMOKE_PASS', len(selected), flush=True)
            values = torch.stack([action_metrics(estimated(delta), simulated(delta))
                                  for delta in perturbations(y)], dim=1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Finite action response metrics required')
            chunks.append(values.cpu())
            if (start + len(selected)) % 192 == 0 or start + len(selected) == len(ids):
                print('TRUE_ESTIMATED_ACTION_PROGRESS', start + len(selected), len(ids), flush=True)
    values = torch.cat(chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete 48-record source receiver/view groups required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[group + '/' + view] = dict(count=len(cell), mean_metrics=cell.mean(0).tolist(),
                                              median_metrics=cell.median(0).values.tolist())
    core.write(args.out / 'result.json', dict(status='TRUE_ESTIMATED_ACTION_SOURCE_COMPLETE',
        commit=args.commit, records=len(ids), directions=list(DIRECTIONS), epsilon_relative_rms=EPSILON,
        metric_order=['relative_squared_difference','real_alignment','estimated_to_true_norm'],
        groups=groups, elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0, DURA_frozen=True,
        caveat='Simulator action is privileged source-only; artificial perturbations do not measure TX identity or prove deployable accuracy.'))
    print('TRUE_ESTIMATED_ACTION_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
