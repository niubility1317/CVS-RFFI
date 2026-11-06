"""Source-only DURA context-variance intervention oracle; never used in target inference."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import iq_nmse

FACTORS = (.125, .25, .5, 1., 2., 4., 8.)


def summarize_factors(values):
    if values.ndim != 2 or values.shape[1] != len(FACTORS) or len(values) == 0 or not torch.isfinite(values).all():
        raise ValueError('Finite factor-wise source NMSE required')
    baseline = values[:, FACTORS.index(1.)]
    oracle = values.min(1).values
    means = values.mean(0)
    return dict(count=len(values), mean_nmse=means.tolist(), baseline_mean=float(baseline.mean()),
                per_record_oracle_mean=float(oracle.mean()), oracle_improved_count=int((oracle < baseline).sum()),
                best_fixed_factor=FACTORS[int(means.argmin())],
                best_factor_counts=torch.bincount(values.argmin(1), minlength=len(FACTORS)).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1 or not torch.cuda.is_available():
        raise ValueError('Exactly one CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    dura = frozen_dura(args.root, 'cuda:0')
    chunks = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start+32]
            x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
            base, action, variance = dura.estimate_context(x)
            if not torch.isfinite(variance).all() or (variance <= 0).any():
                raise FloatingPointError('Positive finite estimated variance required')
            outputs = [dura.stage_outputs(x, (base, action, variance*factor))[0][-1] for factor in FACTORS]
            if start == 0:
                torch.testing.assert_close(outputs[FACTORS.index(1.)], dura(x), rtol=0, atol=0)
                args.out.mkdir(parents=True)
                print('VARIANCE_FACTOR_SOURCE_SMOKE_PASS', flush=True)
            errors = torch.stack([iq_nmse(out, y) for out in outputs], dim=1)
            if not torch.isfinite(errors).all():
                raise FloatingPointError('Finite source IQ errors required')
            chunks.append(errors.cpu())
    values = torch.cat(chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver/view groups required')
            groups[group+'/'+view] = summarize_factors(values[torch.tensor(mask.tolist(), dtype=torch.bool)])
    core.write(args.out/'result.json', dict(status='VARIANCE_FACTOR_SOURCE_COMPLETE', commit=args.commit,
        factors=FACTORS, records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-started,
        model_updates=0, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_IQ_reads=0, target_truth_reads=0,
        caveat='Context variance affects both prior conditioning and MAP/CG. Factor oracle uses same-record source clean IQ and is not deployable; finite factor family only, not identity accuracy or universal upper bound.'))
    print('VARIANCE_FACTOR_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
