"""Source-only factorial swap of original/candidate DURA context and prior."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.action_holdout_source_probe import load_candidate


def mixed_outputs(original, candidate, x):
    a, b = original.estimate_context(x), candidate.estimate_context(x)
    return tuple(model.stage_outputs(x, context)[0][-1]
                 for context, model in ((a, original), (a, candidate), (b, original), (b, candidate)))


def summarize(values):
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0 or not torch.isfinite(values).all():
        raise ValueError('Finite 4-arm per-record source metrics required')
    return dict(count=len(values), mean_nmse=values.mean(0).tolist(),
                improved_vs_original=(values < values[:, :1]).sum(0).tolist())


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
    original = frozen_dura(args.root, 'cuda:0').eval()
    candidate = load_candidate(args.root, 'cuda:0').eval()
    chunks = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start+32]
            x, target = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
            result = mixed_outputs(original, candidate, x)
            if start == 0:
                torch.testing.assert_close(result[0], original(x), rtol=0, atol=0)
                torch.testing.assert_close(result[3], candidate(x), rtol=0, atol=0)
            energy = target.square().mean((1, 2)).clamp_min(1e-8)
            values = torch.stack([((value-target).square().mean((1, 2))/energy) for value in result], 1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Finite paired source metrics required')
            if start == 0:
                args.out.mkdir(parents=True)
                print('CONTEXT_PRIOR_SOURCE_SMOKE_PASS', flush=True)
            chunks.append(values.cpu())
    values = torch.cat(chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver/view cells required')
            groups[group+'/'+view] = summarize(values[torch.tensor(mask.tolist(), dtype=torch.bool)])
    core.write(args.out/'result.json', dict(status='CONTEXT_PRIOR_SOURCE_COMPLETE', commit=args.commit,
        order=['original_context_original_prior', 'original_context_candidate_prior',
               'candidate_context_original_prior', 'candidate_context_candidate_prior'],
        records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-started,
        model_updates=0, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_IQ_reads=0, target_truth_reads=0,
        caveat='Source-only factorial intervention; context jointly includes coarse base, action, variance; source IQ is not identity accuracy.'))
    print('CONTEXT_PRIOR_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
