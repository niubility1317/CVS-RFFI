"""Build source-only posterior line-oracle IQ targets, never used at inference."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.metrics import oracle_line_probe
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_covariance_posterior import posterior_cg
from experiments.phase1_dura_orthogonal.train_source import load_source_training

COVARIANCE_SOURCE = '20261106-phase1-source-covariance-posterior-s2026092701-r04'
COVARIANCE_COMMIT = '01f04f8881dfbeaaedfe8b60a966baeabe53e78d'
CONTEXT_SOURCE = '20261106-phase1-dura-deterministic-cache-s2026092701-r01'
TEACHER_SCHEMA = 'phase1_source_posterior_positive_teacher_v1'


def source_positive_teacher(anchor, clean, posterior):
    """Source-clean ideal nonnegative line target; never available at inference."""
    direction = posterior-anchor
    probe = oracle_line_probe(anchor, clean, direction, limit=.5)
    scale = anchor.square().mean((1, 2), keepdim=True).sqrt()
    direction_scale = direction.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
    alpha = probe['optimal_alpha'].clamp(0, .5)
    return anchor+alpha[:, None, None]*direction/direction_scale*scale, alpha


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    training_environments('cuda:0', os.environ)
    if args.out.exists():
        raise FileExistsError('Unique source posterior teacher output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, _, _, _, _, _ = source_inputs(args.root)
    _, _, _, fit = load_source_training(args.root, ids)
    context_path = args.root/'runs'/CONTEXT_SOURCE/'source_context.pt'
    data, _ = load_cache(context_path, fit, 32, 'cuda:0', deterministic=True)
    covariance_dir = args.root/'runs'/COVARIANCE_SOURCE
    evidence = core.read(covariance_dir/'result.json')
    expected = dict(status='SOURCE_COVARIANCE_POSTERIOR_COMPLETE', commit=COVARIANCE_COMMIT,
        covariance_fit_ids=len(fit), covariance_shrinkage=.1, cg_steps=160,
        source_blend_oracle=True, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0)
    if any(evidence.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed source-only covariance provenance required')
    state = torch.load(covariance_dir/'source_covariance.pt', map_location='cpu', weights_only=True)
    if (state.get('shrinkage') != .1 or not np.array_equal(state.get('fit_ids', []), fit) or
            tuple(state['covariance'].shape) != (512, 512)):
        raise ValueError('Matching fixed source covariance and fit IDs required')
    covariance = state['covariance'].to('cuda:0')
    dura = frozen_dura(args.root, 'cuda:0')
    teachers, stats = [], []
    started = time.monotonic()
    for offset in range(0, len(data[0]), 16):
        x, anchor, _, cached_v, clean = (item[offset:offset+16].to('cuda:0') for item in data)
        base, action, variance = dura.estimate_context(x)
        posterior, residual = posterior_cg(action, x, covariance, variance, steps=160, rtol=1e-4)
        target, alpha = source_positive_teacher(anchor, clean, posterior)
        probe = oracle_line_probe(anchor, clean, posterior-anchor, limit=.5)
        energy = clean.square().mean((1, 2)).clamp_min(1e-8)
        target_nmse = (target-clean).square().mean((1, 2))/energy
        if offset == 0:
            fresh_anchor = dura.stage_outputs(x, (base, action, variance))[0][-1]
            torch.testing.assert_close(fresh_anchor, anchor, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(variance, cached_v, rtol=1e-4, atol=1e-8)
            torch.testing.assert_close(target_nmse, probe['positive_error'], rtol=1e-5, atol=1e-7)
            print('SOURCE_POSTERIOR_TEACHER_SMOKE_PASS', flush=True)
        if not torch.isfinite(target).all() or not torch.isfinite(residual).all():
            raise FloatingPointError('Finite source-only teacher and CG residual required')
        teachers.append(target.cpu())
        stats.append(torch.stack((probe['anchor_error'], target_nmse, alpha, residual,
                                  (probe['optimal_alpha'] < 0).to(alpha.dtype)), 1).cpu())
        if (offset+len(x)) % 4096 == 0:
            print('SOURCE_POSTERIOR_TEACHER_PROGRESS', offset+len(x), flush=True)
    teacher, stats = torch.cat(teachers), torch.cat(stats)
    contract = dict(schema=TEACHER_SCHEMA, fit_ids=fit.tolist(), epochs=[21, 22, 23],
        covariance_source=COVARIANCE_SOURCE, covariance_commit=COVARIANCE_COMMIT,
        context_source=CONTEXT_SOURCE, cg_steps=160, positive_limit=.5,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    with (args.out/'teacher.pt').open('xb') as handle:
        torch.save(dict(contract=contract, teacher=teacher), handle)
    core.write(args.out/'result.json', dict(status='SOURCE_POSTERIOR_TEACHER_COMPLETE',
        commit=args.commit, contract=contract, exposure=len(teacher),
        baseline_nmse=float(stats[:, 0].mean()), teacher_nmse=float(stats[:, 1].mean()),
        positive_step_fraction=float((stats[:, 2] > 0).float().mean()),
        negative_unclipped_fraction=float(stats[:, 4].mean()),
        cg_mean_relative_residual=float(stats[:, 3].mean()), cg_max_relative_residual=float(stats[:, 3].max()),
        elapsed_seconds=time.monotonic()-started, model_updates=0,
        caveat='Every teacher step uses paired source clean IQ and cannot be computed on target; only a separately trained student may be deployed.'))
    print('SOURCE_POSTERIOR_TEACHER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
