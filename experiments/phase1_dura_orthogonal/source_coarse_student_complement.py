"""Source-IQ-only diagnostic of coarse, direct student, and full DURA complementarity."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_student_gain_probe import MODELS, load_student


def pair_metrics(first, second, clean):
    if first.shape != second.shape or first.shape != clean.shape or first.ndim != 3:
        raise ValueError('Matching source IQ batches required')
    direction = second-first
    numerator = ((clean-first)*direction).sum((1, 2))
    denominator = direction.square().sum((1, 2)).clamp_min(1e-12)
    alpha = (numerator/denominator).clamp(0, 1)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    errors = [((v-clean).square().mean((1, 2))/energy) for v in
              (first, second, first+alpha[:, None, None]*direction)]
    return torch.stack((*errors, alpha), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('A unique source-only output is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    _, source, commit, objective, count = MODELS[1]
    student = load_student(args.root, source, commit, objective, count, ids)
    dura = frozen_dura(args.root, 'cpu')
    chunks = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 16):
            selected = ids[start:start+16]
            x, y = core.tensor(received[selected], 'cpu'), core.tensor(clean[selected], 'cpu')
            context = dura.estimate_context(x)
            coarse = context[0]
            full = dura.stage_outputs(x, context)[0][-1]
            direct = student(x)
            if start == 0:
                torch.testing.assert_close(full, dura(x), rtol=1e-5, atol=3e-6)
                args.out.mkdir(parents=True)
                print('COARSE_STUDENT_SOURCE_SMOKE_PASS', flush=True)
            pairs = torch.stack([pair_metrics(a, b, y) for a,b in
                                 ((coarse, direct), (coarse, full), (direct, full))], 1)
            if not torch.isfinite(pairs).all():
                raise FloatingPointError('Finite source IQ errors required')
            if (pairs[:, :, 2] > torch.minimum(pairs[:, :, 0], pairs[:, :, 1])+1e-6).any():
                raise ValueError('Oracle convex mixture must include both endpoints')
            chunks.append(pairs.cpu())
    values = torch.cat(chunks)
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver/view groups required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[name+'/'+view] = dict(count=len(cell), mean_values=cell.mean(0).tolist(),
                improves_both=(cell[:, :, 2] < torch.minimum(cell[:, :, 0], cell[:, :, 1])-1e-8).sum(0).tolist())
    core.write(args.out/'result.json', dict(status='COARSE_STUDENT_SOURCE_COMPLETE', commit=args.commit,
        records=len(ids), groups=groups, pair_order=['coarse_student', 'coarse_dura', 'student_dura'],
        metric_order=['first_nmse', 'second_nmse', 'clean_oracle_mixture_nmse', 'oracle_second_weight'],
        student_source=source, student_commit=commit, elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Per-record mixture uses source clean IQ and cannot be deployed; IQ NMSE is not identity accuracy or an upper bound.'))
    print('COARSE_STUDENT_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
