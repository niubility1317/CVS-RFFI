"""Source-only lag-correlation and phase-aligned repetition oracle diagnostic."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import SOURCE_PILOT, source_inputs

LAGS = (8, 16, 20, 32, 40, 64)


def complex_iq(iq):
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('Source IQ windows [batch, 2, 256] required')
    return torch.complex(iq[:, 0], iq[:, 1])


def lag_correlation(iq, lag):
    z = complex_iq(iq)
    if not 0 < lag < z.shape[1]:
        raise ValueError('Valid lag required')
    left, right = z[:, :-lag], z[:, lag:]
    cross = (right*left.conj()).sum(1)
    norm = (left.abs().square().sum(1)*right.abs().square().sum(1)).clamp_min(1e-20).sqrt()
    return cross, cross.abs()/norm


def phase_aligned_direction(anchor, lag):
    z = complex_iq(anchor)
    cross, _ = lag_correlation(anchor, lag)
    phase = cross/cross.abs().clamp_min(1e-10)
    phase = torch.where(cross.abs() > 1e-10, phase, torch.ones_like(phase))
    changed = torch.cat((z[:, :lag], phase[:, None]*z[:, :-lag]), dim=1)
    direction = changed-z
    return torch.stack((direction.real, direction.imag), dim=1)


def positive_source_oracle(anchor, clean, direction):
    if anchor.shape != clean.shape or anchor.shape != direction.shape:
        raise ValueError('Matching source IQ required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    denominator = direction.square().sum((1, 2)).clamp_min(1e-20)
    alpha = ((clean-anchor)*direction).sum((1, 2))/denominator
    positive = alpha.clamp(0, .5)
    baseline = (anchor-clean).square().mean((1, 2))/energy
    corrected = (anchor+positive[:, None, None]*direction-clean).square().mean((1, 2))/energy
    return alpha, baseline, corrected


@torch.no_grad()
def evaluate(x, anchor, clean):
    outputs = []
    for lag in LAGS:
        _, c = lag_correlation(clean, lag)
        _, r = lag_correlation(x, lag)
        _, a = lag_correlation(anchor, lag)
        alpha, baseline, corrected = positive_source_oracle(anchor, clean, phase_aligned_direction(anchor, lag))
        outputs.append(torch.stack((c, r, a, alpha, baseline, corrected), dim=1))
    result = torch.stack(outputs, dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Finite source lag metrics required')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        raise ValueError('Registered CUDA device required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    dura = frozen_dura(args.root, args.device)
    def batch(start, end):
        x = core.tensor(received[ids[start:end]], args.device)
        y = core.tensor(clean[ids[start:end]], args.device)
        with torch.no_grad():
            return evaluate(x, dura(x), y).cpu()
    smoke = batch(0, 8)
    if smoke.shape != (8, len(LAGS), 6):
        raise ValueError('Real-checkpoint source smoke required')
    print('REPETITION_SOURCE_SMOKE_PASS', len(smoke), flush=True)
    started = time.monotonic()
    all_values = torch.cat([batch(start, min(start+32, len(ids))) for start in range(0, len(ids), 32)])
    values = np.from_dlpack(all_values.contiguous()).copy()
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            subset = values[np.isin(rx, receivers) & (scene == view)]
            if len(subset) != 48:
                raise ValueError('48 source records per group required')
            groups[group+'/'+view] = {}
            for i, lag in enumerate(LAGS):
                v = subset[:, i, :]
                groups[group+'/'+view][str(lag)] = dict(count=len(v), clean_correlation=float(v[:, 0].mean()),
                    received_correlation=float(v[:, 1].mean()), dura_correlation=float(v[:, 2].mean()),
                    negative_optimum_fraction=float((v[:, 3] < 0).mean()),
                    anchor_nmse=float(v[:, 4].mean()), source_oracle_nmse=float(v[:, 5].mean()),
                    improved_fraction=float((v[:, 5] < v[:, 4]-1e-9).mean()))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_REPETITION_DIAGNOSTIC_COMPLETE', commit=args.commit,
        records=len(ids), lags=list(LAGS), groups=groups, elapsed_seconds=time.monotonic()-started,
        source_ids_from=SOURCE_PILOT, F_weights_read=0, F_outputs_read=0, TX_labels_read=0,
        V_reads=0, target_reads=0, model_updates=0,
        caveat='Lag phase comes only from DURA output; step uses source clean and is unavailable at deployment. IQ error is not identity accuracy.'))
    print('REPETITION_SOURCE_DIAGNOSTIC_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
