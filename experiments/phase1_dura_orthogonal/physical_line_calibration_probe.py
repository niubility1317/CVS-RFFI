"""Cross-fitted source-only diagnostic of conservative physical line steps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.physical_line_source_probe import load_model, SOURCE, SOURCE_COMMIT
from experiments.phase1_dura_orthogonal.train_source import extract_context

PHYSICAL_SOURCE = '20261108-phase1-physical-line-source-s2026092701-r01'
PHYSICAL_COMMIT = '0a3a775014a3f8218e41937c4fc33dea1bafd73b'


def fit_beta(q, g, physical):
    remaining = 1 - physical
    denominator = np.sum(q * remaining * remaining)
    if denominator <= 1e-16:
        return 0.
    return float(np.clip(np.sum(remaining * (g - q * physical)) / denominator, 0., 1.))


def iq_error(e0, q, g, alpha):
    return e0 - 2*g*alpha + q*alpha*alpha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    if len(ids) != 672 or len(np.unique(ids)) != 672:
        raise ValueError('672 distinct source records required')
    path = args.root/'runs'/PHYSICAL_SOURCE
    evidence = json.loads((path/'result.json').read_text(encoding='utf8'))
    if (evidence.get('status') != 'PHYSICAL_LINE_SOURCE_COMPLETE' or evidence.get('commit') != PHYSICAL_COMMIT
            or evidence.get('source') != SOURCE or evidence.get('source_commit') != SOURCE_COMMIT
            or any(evidence.get(k) != 0 for k in ('TX_labels_read','F_weights_read','F_outputs_read','V_reads','target_reads','model_updates'))):
        raise ValueError('Matching fixed source physical-line diagnostic required')
    with np.load(path/'values.npz', allow_pickle=False) as z:
        if not np.array_equal(ids, z['ids']) or not np.array_equal(rx, z['rx']) or not np.array_equal(scene, z['scene']):
            raise ValueError('Matching source records and metadata required')
        saved = z['values'].copy()
    model = load_model(args.root, ids)
    dura = frozen_dura(args.root, 'cpu')
    rows = []
    with torch.no_grad():
        for start in range(0, len(ids), 8):
            selected = ids[start:start+8]
            x, clean_iq = core.tensor(received[selected], 'cpu'), core.tensor(clean[selected], 'cpu')
            anchor, innovation, variance = extract_context(dura, x)
            candidate = model(x, anchor, innovation, variance, validate=False)
            delta = candidate - anchor
            residual = clean_iq - anchor
            energy = clean_iq.square().sum((1, 2)).clamp_min(1e-12)
            e0 = residual.square().sum((1, 2)) / energy
            q = delta.square().sum((1, 2)) / energy
            g = (residual*delta).sum((1, 2)) / energy
            if start == 0:
                torch.testing.assert_close(anchor, dura(x), rtol=1e-5, atol=3e-6)
                print('PHYSICAL_CALIBRATION_SOURCE_SMOKE_PASS', len(x), flush=True)
            rows.append(torch.stack((e0, q, g), 1).cpu())
    numbers = np.from_dlpack(torch.cat(rows).contiguous()).copy().astype(np.float64)
    e0, q, g = numbers.T
    if not np.allclose(e0, saved[:,0], rtol=1e-4, atol=2e-7) or not np.allclose(iq_error(e0,q,g,1.), saved[:,1], rtol=1e-4, atol=2e-7):
        raise ValueError('Source waveform replay differs from fixed physical diagnostic')
    physical = saved[:,4].astype(np.float64)
    damaged = scene != 'clean'
    fold = np.full(len(ids), -1, dtype=np.int64)
    for receivers in ((4,6),(8,)):
        for view in core.VIEWS[1:]:
            pos = np.flatnonzero(np.isin(rx,receivers) & (scene==view))
            if len(pos) != 48:
                raise ValueError('48 records per damaged group required')
            fold[pos[::2]] = 0
            fold[pos[1::2]] = 1
    if np.any(fold[damaged] < 0):
        raise ValueError('Complete disjoint source folds required')
    output = np.full(len(ids), np.nan, dtype=np.float64)
    betas = []
    for held in (0, 1):
        fit, test = fold == 1-held, fold == held
        if int(fit.sum()) != 288 or int(test.sum()) != 288:
            raise ValueError('Two disjoint 288-record source folds required')
        beta = fit_beta(q[fit],g[fit],physical[fit])
        output[test] = physical[test] + beta*(1-physical[test])
        betas.append(beta)
    metrics = np.column_stack((e0, iq_error(e0,q,g,1.), iq_error(e0,q,g,physical), iq_error(e0,q,g,output)))
    if not np.isfinite(metrics[damaged]).all():
        raise FloatingPointError('Finite held-out source metrics required')
    groups={}
    for name, receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS[1:]:
            mask = np.isin(rx,receivers) & (scene==view)
            cell = metrics[mask]
            groups[name+'/'+view] = dict(count=len(cell),mean_nmse=cell.mean(0).tolist(),
                calibrated_better_than_physical=int((cell[:,3] < cell[:,2]).sum()),
                calibrated_worse_than_physical=int((cell[:,3] > cell[:,2]).sum()))
    args.out.mkdir(parents=True)
    result=dict(status='PHYSICAL_CALIBRATION_SOURCE_COMPLETE',commit=args.commit,source=SOURCE,
        physical_source=PHYSICAL_SOURCE,records=672,damaged_records=576,fold_sizes=[288,288],
        fit_beta_by_heldout_fold=betas,metric_order=['dura','full_correction','physical','cross_fitted_blend'],
        groups=groups,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='Previously explored source pilot; fold-specific fit uses source clean only for this analysis. Not deployed, no model saved, not TX accuracy or blind target evaluation.')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('PHYSICAL_CALIBRATION_SOURCE_COMPLETE',int(damaged.sum()),flush=True)


if __name__ == '__main__':
    main()
