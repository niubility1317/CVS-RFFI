"""Measure physical innovation alignment on fixed Phase1 source IQ only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.metrics import oracle_line_probe

SOURCE_PILOT = '20261103-phase1-dura-crossfit-blend-source-s2026092701-r01'
SOURCE_COMMIT = '0b93b9aaa229ceb60db81b0028b4564ded3cd57d'


def source_inputs(root):
    """Reuse source physical IDs only; never read pilot predictions or labels."""
    folder = root / 'runs' / SOURCE_PILOT
    meta = core.read(folder / 'prediction_complete.json')
    if any(meta.get(k) != v for k, v in dict(status='SOURCE_PREDICTIONS_FIXED',
            commit=SOURCE_COMMIT, count=672, TX_labels_read=0, U_s_truth_reads=0,
            V_reads=0, test_reads=0).items()):
        raise ValueError('The fixed source IQ-only pilot ID set is required')
    with np.load(folder / 'source_predictions.npz', allow_pickle=False) as z:
        ids, rx, scene, split = (z[k].copy() for k in ('indices', 'rx', 'scene', 'split'))
    if any(v.shape != (672,) for v in (ids, rx, scene, split)) or len(np.unique(ids)) != 672:
        raise ValueError('672 distinct fixed source records required')
    complete = core.read(root / 'runs' / core.F_RUN / 'F' / core.F_ROW / 'source/complete.json')
    contract = core.read(complete['config']['source_contract'])
    cache, clean, roles = core.data(root, contract)
    if (ids.min() < 0 or ids.max() >= len(roles) or
            not np.isin(roles[ids], ('L_s', 'U_s')).all()):
        raise ValueError('Only authorized source L_s/U_s IQ may be used')
    actual_rx = np.load(cache / 'train/rx.npy', allow_pickle=False)
    epoch = cache / 'epochs/19'
    received = np.load(epoch / 'received.npy', mmap_mode='r', allow_pickle=False)
    with np.load(epoch / 'groups.npz', allow_pickle=False) as z:
        actual_scene = z['scene'].copy()
    if (received.shape != clean.shape or received.dtype != np.float32 or
            not np.array_equal(actual_rx[ids], rx) or not np.array_equal(actual_scene[ids], scene)):
        raise ValueError('Source physical IDs, receivers and seven scenes must match')
    return ids, rx, scene, split, received, clean


@torch.no_grad()
def physical_direction(dura, received):
    context = dura.estimate_context(received)
    stages, _, action, _ = dura.stage_outputs(received, context)
    anchor = stages[-1]
    y = torch.complex(received[:, 0], received[:, 1])
    z = torch.complex(anchor[:, 0], anchor[:, 1])
    residual = action.adjoint(y - action(z))
    return anchor, torch.stack((residual.real, residual.imag), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('A unique source diagnostic output is required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, split, received, clean = source_inputs(args.root)
    dura = frozen_dura(args.root, 'cpu')
    x = core.tensor(received[ids[:2]], 'cpu')
    with torch.no_grad():
        smoke_anchor, direction = physical_direction(dura, x)
        if not torch.equal(smoke_anchor, dura(x)):
            raise ValueError('Physical context route must reproduce the fixed DURA waveform')
        smoke = oracle_line_probe(smoke_anchor, core.tensor(clean[ids[:2]], 'cpu'), direction)
    if not all(torch.isfinite(v).all() for v in smoke.values()):
        raise ValueError('Finite real-checkpoint source smoke required')
    print('INNOVATION_SOURCE_SMOKE_PASS', len(x), flush=True)

    chunks = {}
    start_time = time.monotonic()
    for start in range(0, len(ids), 8):
        selected = ids[start:start+8]
        x = core.tensor(received[selected], 'cpu')
        y = core.tensor(clean[selected], 'cpu')
        anchor, direction = physical_direction(dura, x)
        values = oracle_line_probe(anchor, y, direction)
        for name, tensor in values.items():
            chunks.setdefault(name, []).append(tensor.cpu())
        if (start + len(selected)) % 168 == 0:
            print('INNOVATION_SOURCE_PROGRESS', start + len(selected), flush=True)
    arrays = {key: np.from_dlpack(torch.cat(value).contiguous().to(torch.float64)).copy()
              for key, value in chunks.items()}
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Each fixed source receiver group and view must contain 48 records')
            a, p, s = (arrays[k][mask] for k in ('anchor_error', 'positive_error', 'signed_error'))
            alpha = arrays['optimal_alpha'][mask]
            groups[group + '/' + view] = dict(count=int(mask.sum()), anchor_nmse=float(a.mean()),
                oracle_positive_nmse=float(p.mean()), oracle_signed_nmse=float(s.mean()),
                beneficial_positive_fraction=float(np.mean(p < a - 1e-9)),
                negative_optimum_fraction=float(np.mean(alpha < -1e-8)),
                clipped_fraction=float(np.mean(np.abs(alpha) > .5)),
                mean_optimal_alpha=float(alpha.mean()))
    report = dict(status='SOURCE_ORACLE_DIAGNOSTIC_COMPLETE', commit=args.commit, records=len(ids),
                  groups=groups, elapsed_seconds=time.monotonic()-start_time, source_only=True,
                  source_ids_from=SOURCE_PILOT, source_pilot_predictions_read=False,
                  TX_labels_read=0, U_s_truth_reads=0, F_weights_read=0, F_outputs_read=0,
                  V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
                  caveat='Uses source clean IQ to choose an ideal per-record coefficient; not deployable, not an identity metric, and not a certified upper bound.')
    core.write(args.out/'result.json', report)
    print('INNOVATION_SOURCE_COMPLETE', json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
