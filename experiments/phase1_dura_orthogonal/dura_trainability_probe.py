"""One source-only technical gradient step through the original DURA graph."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs


def gradient_summary(model):
    groups = {}
    for name, parameter in model.named_parameters():
        group = groups.setdefault(name.split('.')[0], {'parameters': 0, 'with_gradient': 0,
                                                       'nonzero_gradient': 0, 'gradient_finite': True})
        group['parameters'] += parameter.numel()
        if parameter.grad is not None:
            group['with_gradient'] += parameter.numel()
            group['nonzero_gradient'] += int(torch.count_nonzero(parameter.grad))
            group['gradient_finite'] &= bool(torch.isfinite(parameter.grad).all())
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('A unique source technical output is required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or not torch.cuda.is_available():
        raise ValueError('Exactly one GPU must be selected')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    positions = []
    for view in core.VIEWS[:4]:
        cell = np.flatnonzero(np.isin(rx, (4, 6)) & (scene == view))
        if len(cell) != 48:
            raise ValueError('Complete source receiver/view cells required')
        positions.append(int(cell[0]))
    selected = ids[np.asarray(positions)]
    model = frozen_dura(args.root, 'cuda:0').requires_grad_(True).eval()
    x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    output = model(x)
    if output.shape != x.shape or not torch.isfinite(output).all():
        raise FloatingPointError('Finite original DURA output required')
    energy = y.square().mean((1, 2)).clamp_min(1e-8)
    loss = ((output-y).square().mean((1, 2))/energy).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Finite source-only IQ loss required')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    groups = gradient_summary(model)
    if not groups or not all(g['gradient_finite'] for g in groups.values()) or not any(g['nonzero_gradient'] for g in groups.values()):
        raise FloatingPointError('Finite nonzero physical graph gradients required')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    before = next(p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad)).detach().clone()
    chosen = next(p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad))
    optimizer.step()
    if torch.equal(before, chosen.detach()) or not torch.isfinite(chosen).all():
        raise ValueError('The in-memory source-only technical step must update a finite parameter')
    torch.cuda.synchronize()
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='DURA_TRAINABILITY_SOURCE_PASS', commit=args.commit,
        source_records=len(selected), views=list(core.VIEWS[:4]), gpu_device='cuda:0',
        gradient_groups=groups, source_iq_loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), technical_optimizer_steps=1,
        weights_saved=False, original_checkpoint_modified=False, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0,
        caveat='Transient technical step only; no new model or identity performance result.'))
    print('DURA_TRAINABILITY_SOURCE_PASS', flush=True)


if __name__ == '__main__':
    main()
