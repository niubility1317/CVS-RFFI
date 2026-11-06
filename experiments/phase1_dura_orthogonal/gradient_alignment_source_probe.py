"""Compare source-only IQ and physics gradient directions without model updates."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.action_holdout_source_probe import load_candidate, TRAIN_PROBE_SEED
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
from experiments.phase1_dura_orthogonal.train_source import action_response_loss, source_loss


def alignment(iq, action):
    if len(iq) != len(action):
        raise ValueError('Matching gradient parameter lists required')
    pairs = [(a, b) for a, b in zip(iq, action) if a is not None and b is not None]
    if not pairs:
        raise ValueError('Shared gradient parameters required')
    x = torch.cat([a.detach().reshape(-1).double() for a, _ in pairs])
    y = torch.cat([b.detach().reshape(-1).double() for _, b in pairs])
    nx, ny = x.norm(), y.norm()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or nx <= 0 or ny <= 0:
        raise ValueError('Finite nonzero shared gradients required')
    return dict(shared_tensors=len(pairs), shared_elements=x.numel(),
                cosine=float(torch.dot(x, y)/(nx*ny)), iq_norm=float(nx),
                action_norm=float(ny), weighted_action_to_iq_norm=float(.01*ny/nx))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique output required')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1 or not torch.cuda.is_available():
        raise ValueError('Exactly one CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    folder = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/epochs/19'
    meta = core.read(folder/'complete.json')
    if (meta.get('count') != core.COUNT or meta.get('namespace') != 'learned-action/train/epoch19'
            or any(meta.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
        raise ValueError('Approved matching source cache required')
    kernel = np.load(folder/'kernel.npy', mmap_mode='r', allow_pickle=False)
    coeff = np.load(folder/'coefficients.npy', mmap_mode='r', allow_pickle=False)
    if kernel.shape != (core.COUNT, 2, 64) or coeff.shape != (core.COUNT, 4, 256):
        raise ValueError('Complete source action required')
    original = frozen_dura(args.root, 'cuda:0').eval()
    candidate = load_candidate(args.root, 'cuda:0').eval()
    probe_parts = torch.randn((1, 2, 256), generator=torch.Generator().manual_seed(TRAIN_PROBE_SEED)).to('cuda:0')
    probe = torch.complex(probe_parts[:, 0], probe_parts[:, 1])
    groups = {}
    started = time.monotonic()
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            cell = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(cell) != 48:
                raise ValueError('Complete source receiver/view cells required')
            selected = ids[cell[:16]]
            x, clean_x = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
            h, c = core.tensor(kernel[selected], 'cuda:0'), core.tensor(coeff[selected], 'cuda:0')
            with torch.no_grad():
                anchor = original(x).detach()
            results = {}
            for label, model in (('original', original), ('candidate', candidate)):
                model.requires_grad_(True)
                params = tuple(model.parameters())
                context = model.estimate_context(x)
                output = model.stage_outputs(x, context)[0][-1]
                ref = true_action(type(context[1]), h, c)
                iq_loss = source_loss(output, anchor, clean_x, objective='dura_guard')
                physics_loss = action_response_loss(context[1], ref, probe.expand(len(x), -1))
                iq_grads = torch.autograd.grad(iq_loss, params, retain_graph=True, allow_unused=True)
                phys_grads = torch.autograd.grad(physics_loss, params, allow_unused=True)
                result = alignment(iq_grads, phys_grads)
                result.update(iq_loss=float(iq_loss.detach()), physics_loss=float(physics_loss.detach()))
                results[label] = result
                model.requires_grad_(False)
            groups[group+'/'+view] = dict(count=len(selected), **results)
            if len(groups) == 1:
                args.out.mkdir(parents=True)
                print('GRADIENT_ALIGNMENT_SOURCE_SMOKE_PASS', flush=True)
    core.write(args.out/'result.json', dict(status='GRADIENT_ALIGNMENT_SOURCE_COMPLETE', commit=args.commit,
        groups=groups, records_per_group=16, probe_seed=TRAIN_PROBE_SEED,
        elapsed_seconds=time.monotonic()-started, model_updates=0, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0,
        caveat='Source pilot batch-level gradient comparison, shared parameters only; no identity accuracy or training guarantee.'))
    print('GRADIENT_ALIGNMENT_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
