"""Source-only oracle phase/gain decomposition; never used at inference."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.adaptive_inverse import AdaptiveInverseRecovery
from experiments.phase1_dura_orthogonal.direct_source_probe import load_direct
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, DUAL_DOMAIN_EPOCHS, FIT_SIZE

SOURCE = '20261106-phase1-adaptive-inverse-source-s2026092701-r01'
SOURCE_COMMIT = 'c406307494b30bd5bdcc6d619337a5250c8d46e9'


def gain_metrics(prediction, clean):
    if prediction.shape != clean.shape or prediction.ndim != 3 or prediction.shape[1] != 2:
        raise ValueError('Matching IQ batches required')
    x = torch.complex(prediction[:, 0], prediction[:, 1])
    y = torch.complex(clean[:, 0], clean[:, 1])
    reference_energy = y.abs().square().sum(-1).clamp_min(1e-12)
    inner = (x.conj()*y).sum(-1)
    energy = x.abs().square().sum(-1)
    magnitude = inner.abs()
    phase = torch.where(magnitude > 1e-20, inner/magnitude.clamp_min(1e-20), torch.ones_like(inner))
    gain = inner/energy.clamp_min(1e-20)
    return torch.stack(tuple((scale[:, None]*x-y).abs().square().sum(-1)/reference_energy
                       for scale in (torch.ones_like(inner), phase, gain)), dim=1)


def load_adaptive(root, device):
    folder = root/'runs'/SOURCE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=SOURCE_COMMIT, architecture='adaptive_inverse', objective='iq',
                    epochs=DUAL_DOMAIN_EPOCHS, adaptive_filter_taps=33, DURA_training_reads=0,
                    TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                    target_reads=0, DURA_frozen=True)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != FIT_SIZE or
            state.get('schema') != SCHEMA or len(state.get('fit_ids', [])) != FIT_SIZE or
            any(summary.get(k) != v or state.get(k) != v for k, v in expected.items())):
        raise ValueError('Fixed source-only adaptive inverse checkpoint required')
    model = AdaptiveInverseRecovery(width=state['width'], taps=33)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1 or not torch.cuda.is_available():
        raise ValueError('Exactly one CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    adaptive, direct, dura = load_adaptive(args.root, 'cuda:0'), load_direct(args.root, 'cuda:0'), frozen_dura(args.root, 'cuda:0')
    values = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            chosen = ids[start:start+32]
            x, y = core.tensor(received[chosen], 'cuda:0'), core.tensor(clean[chosen], 'cuda:0')
            result = torch.stack([gain_metrics(v, y) for v in (x, adaptive(x), direct(x), dura(x))], dim=1)
            if not torch.isfinite(result).all():
                raise FloatingPointError('Finite source oracle metrics required')
            if (result[:, :, 2] > result[:, :, 1]+1e-5).any() or (result[:, :, 1] > result[:, :, 0]+1e-5).any():
                raise ValueError('Oracle nested optimum ordering must hold')
            if start == 0:
                args.out.mkdir(parents=True)
                print('COMPLEX_GAIN_SOURCE_SMOKE_PASS', flush=True)
            values.append(result.cpu())
    values = torch.cat(values)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver/view cells required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[group+'/'+view] = dict(count=len(cell), mean_nmse=cell.mean(0).tolist())
    core.write(args.out/'result.json', dict(status='COMPLEX_GAIN_SOURCE_COMPLETE', commit=args.commit,
        model_order=['received', 'adaptive_inverse', 'dual_domain', 'dura'],
        metric_order=['unaligned', 'oracle_phase_only', 'oracle_complex_gain'],
        records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-started, model_updates=0,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0,
        caveat='Per-record clean-IQ oracle alignment is privileged source diagnosis, not a deployable recovery or identity result.'))
    print('COMPLEX_GAIN_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
