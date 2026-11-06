"""Source-only separability of DURA observable noise statistic by scene."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.train_source import extract_context


def pair_auc(clean, practical):
    if (clean.ndim != 1 or practical.ndim != 1 or len(clean) == 0 or len(practical) == 0
            or not torch.isfinite(clean).all() or not torch.isfinite(practical).all()):
        raise ValueError('Nonempty finite source statistic groups required')
    left, right = practical[:, None], clean[None, :]
    return float(((left > right).float() + .5*(left == right).float()).mean())


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
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    dura = frozen_dura(args.root, 'cuda:0')
    values = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            x = core.tensor(received[ids[start:start+32]], 'cuda:0')
            anchor, _, variance = extract_context(dura, x)
            signal_power = anchor.square().mean((1, 2)).clamp_min(1e-12)
            statistic = (variance/signal_power).clamp_min(1e-12).log10()
            if not torch.isfinite(statistic).all():
                raise FloatingPointError('Finite normalized variance required')
            if start == 0:
                args.out.mkdir(parents=True)
                print('VARIANCE_SEPARABILITY_SOURCE_SMOKE_PASS', flush=True)
            values.append(statistic.cpu())
    values = torch.cat(values)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        clean_mask = np.isin(rx, receivers) & (scene == 'clean')
        if clean_mask.sum() != 48:
            raise ValueError('Complete source clean group required')
        clean = values[torch.tensor(clean_mask.tolist(), dtype=torch.bool)]
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source scene group required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[group+'/'+view] = dict(count=len(cell), quantiles_05_50_95=
                torch.quantile(cell, torch.tensor([.05, .5, .95])).tolist(),
                auc_vs_clean=None if view == 'clean' else pair_auc(clean, cell))
    core.write(args.out/'result.json', dict(status='VARIANCE_SEPARABILITY_SOURCE_COMPLETE', commit=args.commit,
        statistic='log10(DURA estimated variance / DURA anchor real-IQ mean square)',
        records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-started, model_updates=0,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0,
        caveat='Scene is source simulator metadata used for retrospective descriptive comparison only. AUC is not deployment accuracy or a selected gate threshold.'))
    print('VARIANCE_SEPARABILITY_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
