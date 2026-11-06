"""Source-only cross-fitted complex FIR error decomposition for fixed students."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_student_gain_probe import MODELS, load_student

TAPS = 17
RIDGE = 1e-3
HALVES = ((8, 128, 128, 248), (128, 248, 8, 128))


def windows(signal):
    if signal.ndim != 3 or signal.shape[1:] != (2, 256):
        raise ValueError('Matching [batch,2,256] source IQ required')
    z = torch.complex(signal[:, 0].double(), signal[:, 1].double())
    return F.pad(z, (TAPS//2, TAPS//2)).unfold(1, TAPS, 1), z


def fit_filter(matrix, target, begin, end):
    x, y = matrix[:, begin:end], target[:, begin:end]
    gram = x.conj().transpose(1, 2) @ x
    rhs = x.conj().transpose(1, 2) @ y.unsqueeze(-1)
    scale = gram.diagonal(dim1=1, dim2=2).real.mean(1).clamp_min(1e-12)
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.solve(gram + RIDGE*scale[:, None, None]*eye, rhs).squeeze(-1)


def crossfit_metrics(prediction, clean):
    if prediction.shape != clean.shape:
        raise ValueError('Paired source IQ required')
    matrix, z = windows(prediction)
    y = torch.complex(clean[:, 0].double(), clean[:, 1].double())
    outputs = []
    for begin, end, evaluate, stop in HALVES:
        weight = fit_filter(matrix, y, begin, end)
        calibrated = (matrix[:, evaluate:stop] @ weight.unsqueeze(-1)).squeeze(-1)
        x_train, y_train = z[:, begin:end], y[:, begin:end]
        scalar = (x_train.conj()*y_train).sum(1)/x_train.abs().square().sum(1).clamp_min(1e-20)
        held_x, held_y = z[:, evaluate:stop], y[:, evaluate:stop]
        energy = held_y.abs().square().sum(1).clamp_min(1e-12)
        outputs.append(torch.stack(tuple((guess-held_y).abs().square().sum(1)/energy
            for guess in (held_x, scalar[:, None]*held_x, calibrated)), 1))
    return torch.stack(outputs, 1).mean(1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    models = [load_student(args.root, run, commit, objective, count, ids)
              for _, run, commit, objective, count in MODELS]
    dura = frozen_dura(args.root, 'cpu')
    chunks = []
    started = time.monotonic()
    for offset in range(0, len(ids), 16):
        selected = ids[offset:offset+16]
        x, y = core.tensor(received[selected], 'cpu'), core.tensor(clean[selected], 'cpu')
        metrics = torch.stack([crossfit_metrics(v, y) for v in (x, models[0](x), models[1](x), dura(x))], 1)
        if not torch.isfinite(metrics).all():
            raise FloatingPointError('Finite cross-fitted source metrics required')
        if offset == 0:
            print('STUDENT_CROSSFIT_FILTER_SMOKE_PASS', len(selected), flush=True)
        chunks.append(metrics)
    values = torch.cat(chunks)
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('48 records per source receiver/view required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[name+'/'+view] = dict(count=len(cell), mean_nmse=cell.mean(0).tolist(),
                filter_better_than_raw=(cell[:, :, 2] < cell[:, :, 0]).sum(0).tolist())
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='STUDENT_CROSSFIT_FILTER_SOURCE_COMPLETE', commit=args.commit,
        model_order=['received', 'student_8192', 'student_32768', 'dura'],
        metric_order=['unaligned', 'crossfit_scalar', 'crossfit_complex_fir17'], taps=TAPS, ridge=RIDGE,
        halves=[list(v) for v in HALVES], records=len(ids), groups=groups,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, model_updates=0,
        caveat='Each filter fits a clean source half and is evaluated on the other; clean calibration is unavailable at deployment. Not identity accuracy or a general bound.'))
    print('STUDENT_CROSSFIT_FILTER_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
