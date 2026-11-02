"""Source-only fixed-grid waveform fusion diagnostic; no TX truth is read."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura

PILOT = '20261102-phase1-hankel-posterior-source-s2026092701-r01'
ALPHAS = (0., .25, .5, .75, 1.)


def comparison(classes, pseudo, confidence):
    trusted = confidence >= .95
    matches = classes[trusted] == pseudo[trusted, None]
    endpoints = matches[:, 0] | matches[:, -1]
    interior = matches[:, 1:-1].any(axis=1)
    return dict(count=len(classes), trusted=int(trusted.sum()),
                match_by_alpha=matches.sum(axis=0).astype(int).tolist(),
                endpoint_union=int(endpoints.sum()), all_alpha_union=int(matches.any(axis=1).sum()),
                interior_only=int((interior & ~endpoints).sum()))


def predict(root, out, commit, smoke):
    from experiments.phase1_dura_posterior.physical_source_smoke import load_source_models, s
    if out.exists(): raise FileExistsError('Unique prediction output required')
    pilot = root/'runs'/PILOT
    manifest = core.read(pilot/'prediction_complete.json')
    if manifest.get('status') != 'SOURCE_PREDICTIONS_FIXED' or manifest.get('count') != 672:
        raise ValueError('Fixed source pilot cohort required')
    with np.load(pilot/'source_predictions.npz', allow_pickle=False) as z:
        ids, split, saved_rx, saved_scene = (z[k].copy() for k in ('indices', 'split', 'rx', 'scene'))
    cache, clean, rx, training, pv, ph = s.source_contract(root)
    validation, heldout = s.new_splits(cache, rx, training, pv, ph)
    received, scene = s.epoch_data(cache, 19)
    if (len(np.unique(ids)) != 672 or np.intersect1d(ids, training).size
            or not np.isin(ids, np.r_[validation, heldout]).all()
            or not np.isin(ids[split == 'RX4_6'], validation).all()
            or not np.isin(ids[split == 'RX8'], heldout).all()
            or not np.array_equal(saved_rx, rx[ids]) or not np.array_equal(saved_scene, scene[ids])):
        raise ValueError('Disjoint, fixed source rows required')
    selected = []
    per_cell = 2 if smoke else 32
    for group in ('RX4_6', 'RX8'):
        for view in core.VIEWS:
            choices = np.flatnonzero((split == group) & (scene[ids] == view))
            if len(choices) < per_cell: raise ValueError('Complete source receiver/view cells required')
            selected.extend(choices[:per_cell])
    selected = np.asarray(selected, dtype=np.int64)
    physical, gate = load_source_models(root, 'mean')
    physical.eval().requires_grad_(False); gate.eval().requires_grad_(False)
    dura = frozen_dura(root, 'cpu'); reader, _ = core.teacher(root, 'cpu')
    reader.eval().requires_grad_(False)
    classifications, errors = [], []
    with torch.no_grad():
        for start in range(0, len(selected), 8):
            ix = ids[selected[start:start+8]]
            x = core.tensor(received[ix], 'cpu'); y = core.tensor(clean[ix], 'cpu')
            active = gate(x).sigmoid().reshape(-1) > .8
            posterior = x.clone()
            if active.any(): posterior[active] = physical(x[active])[0]
            base = dura(x)
            denominator = y[:, :, 64:-64].square().sum(1).mean(1).clamp_min(1e-12)
            class_rows, error_rows = [], []
            for alpha in ALPHAS:
                wave = base + alpha*(posterior-base)
                logits = reader.classify_features(reader.features(wave))
                _, error = s.align(wave, y)
                if not torch.isfinite(logits).all() or not torch.isfinite(error).all():
                    raise FloatingPointError('Finite fused predictions required')
                class_rows.append(logits.argmax(1)); error_rows.append(error/denominator)
            classifications.append(torch.stack(class_rows, 1).cpu())
            errors.append(torch.stack(error_rows, 1).cpu())
            if (start+len(ix)) % 64 == 0:
                print('SOURCE_FUSION_PROGRESS', start+len(ix), flush=True)
    out.mkdir(parents=True)
    with (out/'source_predictions.npz').open('xb') as f:
        np.savez(f, indices=ids[selected], split=split[selected], rx=rx[ids[selected]],
                 scene=scene[ids[selected]], classes=torch.cat(classifications).numpy(),
                 aligned_error=torch.cat(errors).numpy())
    core.write(out/'prediction_complete.json', dict(status='SOURCE_PREDICTIONS_FIXED', commit=commit,
               count=len(selected), alphas=ALPHAS, smoke=smoke, F_frozen=True, DURA_frozen=True,
               TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates=0))
    print('SOURCE_FUSION_FIXED', len(selected), flush=True)


def score(root, source, out, commit):
    from experiments.phase1_dura_posterior.physical_source_smoke import s
    if out.exists(): raise FileExistsError('Unique source score required')
    manifest = core.read(source/'prediction_complete.json')
    if (manifest.get('status') != 'SOURCE_PREDICTIONS_FIXED' or manifest.get('commit') != commit
            or manifest.get('count') != 448 or manifest.get('smoke') is not False):
        raise ValueError('All source predictions must be fixed before scoring')
    with np.load(source/'source_predictions.npz', allow_pickle=False) as z:
        ids, split, scene, classes, errors = (z[k].copy() for k in ('indices', 'split', 'scene', 'classes', 'aligned_error'))
    if len(np.unique(ids)) != 448 or classes.shape != (448, 5) or errors.shape != (448, 5):
        raise ValueError('Complete unique source predictions required')
    cache, clean, *_ = s.source_contract(root)
    reader, _ = core.teacher(root, 'cpu'); reader.eval().requires_grad_(False)
    pseudo, confidence = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 8):
            y = core.tensor(clean[ids[start:start+8]], 'cpu')
            conf, pred = F.softmax(reader.classify_features(reader.features(y)), 1).max(1)
            confidence.append(conf.cpu()); pseudo.append(pred.cpu())
    confidence = torch.cat(confidence).numpy(); pseudo = torch.cat(pseudo).numpy()
    results = {}
    for group in ('RX4_6', 'RX8'):
        results[group] = {}
        for view in core.VIEWS:
            mask = (split == group) & (scene == view)
            if int(mask.sum()) != 32: raise ValueError('Complete source cells required')
            row = comparison(classes[mask], pseudo[mask], confidence[mask])
            row['mean_aligned_error_by_alpha'] = errors[mask].mean(0).tolist()
            results[group][view] = row
    out.parent.mkdir(parents=True, exist_ok=False)
    core.write(out, dict(status='SOURCE_PSEUDO_SCORE_COMPLETE', commit=commit, count=len(ids),
               alphas=ALPHAS, results=results, TX_truth_reads=0, U_s_truth_reads=0,
               V_reads=0, test_reads=0, model_updates=0,
               caveat='Clean-F pseudo targets are not TX truth. Oracle unions are not deployable.'))
    print('SOURCE_FUSION_SCORE_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('predict', 'score'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True); p.add_argument('--source', type=Path); p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); torch.set_num_threads(2)
    if args.mode == 'predict': predict(args.root, args.out, args.commit, args.smoke)
    else:
        if args.source is None or args.smoke: raise ValueError('Fixed full source predictions required')
        score(args.root, args.source, args.out, args.commit)


if __name__ == '__main__': main()
