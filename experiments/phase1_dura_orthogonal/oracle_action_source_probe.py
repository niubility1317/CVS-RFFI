"""Replace DURA context with source simulator action/noise for diagnosis only."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs


def true_action(action_type, kernel, coefficients):
    if (kernel.ndim != 3 or kernel.shape[1:] != (2, 64) or
            coefficients.shape != (len(kernel), 4, 256)):
        raise ValueError('Matched simulator kernel and widely-linear coefficients required')
    if not torch.isfinite(kernel).all() or not torch.isfinite(coefficients).all():
        raise FloatingPointError('Finite source simulator action required')
    h = torch.complex(kernel[:, 0], kernel[:, 1])
    main = torch.complex(coefficients[:, 0], coefficients[:, 1])
    image = torch.complex(coefficients[:, 2], coefficients[:, 3])
    return action_type(h, main, image)


def iq_nmse(output, clean):
    if output.shape != clean.shape or output.ndim != 3 or output.shape[1:] != (2, 256):
        raise ValueError('Matching IQ windows required')
    return (output-clean).square().mean((1, 2))/clean.square().mean((1, 2)).clamp_min(1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or not torch.cuda.is_available():
        raise ValueError('Exactly one GPU must be selected')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    folder = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/epochs/19'
    metadata = core.read(folder/'complete.json')
    if (metadata.get('count') != core.COUNT or metadata.get('namespace') != 'learned-action/train/epoch19'
            or any(metadata.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
        raise ValueError('Approved source simulator action cache required')
    kernel = np.load(folder/'kernel.npy', mmap_mode='r', allow_pickle=False)
    coefficients = np.load(folder/'coefficients.npy', mmap_mode='r', allow_pickle=False)
    variance = np.load(folder/'variance.npy', mmap_mode='r', allow_pickle=False)
    if (kernel.shape != (core.COUNT, 2, 64) or coefficients.shape != (core.COUNT, 4, 256)
            or variance.shape != (core.COUNT,) or any(v.dtype != np.float32 for v in (kernel, coefficients, variance))):
        raise ValueError('Complete matching simulator action arrays required')
    dura = frozen_dura(args.root, 'cuda:0')
    chunks = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start+32]
            x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
            h = core.tensor(kernel[selected], 'cuda:0')
            c = core.tensor(coefficients[selected], 'cuda:0')
            v = core.tensor(variance[selected], 'cuda:0')
            if not torch.isfinite(v).all() or (v <= 0).any():
                raise ValueError('Positive finite simulator variance required')
            base, estimated, estimated_v = dura.estimate_context(x)
            oracle = true_action(type(estimated), h, c)
            contexts = ((base, estimated, estimated_v), (base, estimated, v),
                        (base, oracle, estimated_v), (base, oracle, v))
            outputs = [dura.stage_outputs(x, context)[0][-1] for context in contexts]
            if start == 0:
                torch.testing.assert_close(outputs[0], dura(x), rtol=1e-5, atol=3e-6)
                args.out.mkdir(parents=True)
                print('ORACLE_ACTION_SOURCE_SMOKE_PASS', flush=True)
            errors = torch.stack([iq_nmse(z, y) for z in outputs], dim=1)
            if not torch.isfinite(errors).all():
                raise FloatingPointError('Finite simulator-intervention output required')
            chunks.append(errors.cpu())
    values = torch.cat(chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver and scene cells required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[group+'/'+view] = dict(count=len(cell), mean_nmse=cell.mean(0).tolist(),
                better_than_baseline=(cell[:, 1:] < cell[:, :1]).sum(0).tolist())
    core.write(args.out/'result.json', dict(status='ORACLE_ACTION_SOURCE_COMPLETE', commit=args.commit,
        records=len(ids), order=['estimated_action_estimated_variance', 'estimated_action_true_variance',
                                 'true_action_estimated_variance', 'true_action_true_variance'],
        groups=groups, elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Simulator action and variance are source-only privileged oracles and cannot be used at target inference; IQ NMSE is not identity accuracy.'))
    print('ORACLE_ACTION_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
