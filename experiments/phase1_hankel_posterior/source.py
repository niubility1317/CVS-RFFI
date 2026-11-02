"""Source-only fixed Hankel diagnostic; identity labels are never opened here."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_hankel_posterior.hankel import hankel_shrink


REFERENCE = '20261102-phase1-dura-posterior-internal-source-s2026092701-r01'


def main():
    from experiments.phase1_dura_posterior.physical_source_smoke import load_source_models, s
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique fixed source output is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = args.root/'runs'/REFERENCE
    manifest = core.read(reference/'prediction_complete.json')
    if (manifest.get('status') != 'SOURCE_PREDICTIONS_FIXED' or manifest.get('count') != 672
            or manifest.get('TX_labels_read') != 0 or manifest.get('upstream_L_s_fit_labels_used') != 0):
        raise ValueError('Fixed label-hidden source cohort is required')
    with np.load(reference/'source_predictions.npz', allow_pickle=False) as z:
        ids, split, saved_rx, saved_scene = (z[k].copy() for k in ('indices', 'split', 'rx', 'scene'))
    if ids.shape != (672,) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique fixed source physical records are required')
    cache, clean, rx, training, pv, ph = s.source_contract(args.root)
    validation, heldout = s.new_splits(cache, rx, training, pv, ph)
    received, scenes = s.epoch_data(cache, 19)
    if (np.intersect1d(ids, training).size or not np.all(np.isin(ids, np.r_[validation, heldout]))
            or not np.array_equal(rx[ids], saved_rx) or not np.array_equal(scenes[ids], saved_scene)):
        raise ValueError('Source cohort must match disjoint receiver and view assignments')
    if args.smoke:
        selected = []
        for group in ('RX4_6', 'RX8'):
            for view in core.VIEWS:
                index = np.flatnonzero((split == group) & (saved_scene == view))
                if len(index) < 2:
                    raise ValueError('Two source records per group and view are required')
                selected.extend(index[:2].tolist())
        selected = np.asarray(selected, dtype=np.int64)
    else:
        selected = np.arange(len(ids))
    physical, gate = load_source_models(args.root, 'mean')
    physical.eval().requires_grad_(False)
    gate.eval().requires_grad_(False)
    reader, _ = core.teacher(args.root, 'cpu')
    reader.eval().requires_grad_(False)
    rows = []
    with torch.no_grad():
        for start in range(0, len(selected), 8):
            pos = selected[start:start+8]
            ix = ids[pos]
            x, y = core.tensor(received[ix], 'cpu'), core.tensor(clean[ix], 'cpu')
            active = gate(x).sigmoid() > .8
            parent, output = x.clone(), x.clone()
            if active.any():
                mean, logvar = physical(x[active])
                parent[active] = mean
                output[active] = hankel_shrink(mean, logvar.exp())
            clean_logits = reader.classify_features(reader.features(y))
            confidence, pseudo = F.softmax(clean_logits, dim=1).max(1)
            energy = y[:, :, 64:-64].square().sum(1).mean(1).clamp_min(1e-12)
            fields = [confidence, pseudo.float(), active.float()]
            for wave in (x, parent, output):
                logits = reader.classify_features(reader.features(wave))
                if not torch.isfinite(wave).all() or not torch.isfinite(logits).all():
                    raise FloatingPointError('Finite source waves and reader logits are required')
                _, error = s.align(wave, y)
                fields.extend((logits.argmax(1).float(), error/energy))
            rows.append(torch.stack(fields, dim=1).cpu().contiguous())
            if (start+len(pos)) % 128 == 0:
                print('HANKEL_SOURCE_PROGRESS', start+len(pos), flush=True)
    values = np.from_dlpack(torch.cat(rows).contiguous())
    if values.shape != (len(selected), 9) or not np.isfinite(values).all():
        raise ValueError('Complete finite source predictions are required')
    args.out.mkdir(parents=True)
    if args.smoke:
        core.write(args.out/'result.json', dict(status='PASS', records=len(selected),
                   source_only=True, TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, test_reads=0))
        print('HANKEL_SOURCE_SMOKE_PASS', len(selected), flush=True)
        return
    with (args.out/'source_predictions.npz').open('xb') as f:
        np.savez(f, indices=ids, split=split, rx=saved_rx, scene=saved_scene, values=values)
    core.write(args.out/'prediction_complete.json', dict(status='SOURCE_PREDICTIONS_FIXED',
               commit=args.commit, count=len(ids), source_only=True, DURA_loaded=False,
               F_frozen=True, TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, test_reads=0,
               model_updates=0, window=32, shrink_scale=0.1, mix=0.5,
               columns='clean_confidence,clean_pseudoclass,gate,raw_class,raw_aligned_iq,physical_class,physical_aligned_iq,hankel_class,hankel_aligned_iq',
               caveat='Previously explored source cohort; blind observation variance is a heuristic scale.'))
    print('HANKEL_SOURCE_PREDICTIONS_FIXED', len(ids), flush=True)


if __name__ == '__main__':
    main()
