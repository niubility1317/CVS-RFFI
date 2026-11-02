"""Fixed-parameter Hankel replay on the complete pre-existing source cohort."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_hankel_posterior.hankel import hankel_shrink

PRIOR = '20261102-phase1-unroll-source-replay-s2026092701-r01'
PILOT = '20261102-phase1-hankel-posterior-source-s2026092701-r01'


@torch.no_grad()
def compute(x, y, physical, gate, reader, source):
    scores = gate(x).sigmoid().reshape(-1)
    active = scores > .8
    parent, output = x.clone(), x.clone()
    if active.any():
        mean, logvar = physical(x[active])
        parent[active] = mean
        output[active] = hankel_shrink(mean, logvar.exp())
    predictions = []
    for wave in (x, parent, output):
        logits = reader.classify_features(reader.features(wave))
        if not torch.isfinite(wave).all() or not torch.isfinite(logits).all():
            raise FloatingPointError('Finite source waveforms and classifier outputs are required')
        predictions.append(logits.argmax(1))
    denominator = y[:, :, 64:-64].square().sum(1).mean(1).clamp_min(1e-12)
    _, error = source.align(output, y)
    return torch.stack((predictions[0].float(), active.float(), predictions[1].float(),
                        predictions[2].float(), error/denominator), dim=1)


def main():
    from experiments.phase1_dura_posterior.physical_source_smoke import load_source_models, s
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError('Unique full source replay is required')
    torch.set_num_threads(2)
    prior = args.root/'runs'/PRIOR
    pm = core.read(prior/'prediction_complete.json')
    if (pm.get('status') != 'SOURCE_PREDICTIONS_FIXED' or pm.get('count') != 11761
            or any(pm.get(k) != 0 for k in ('TX_truth_reads', 'U_s_truth_reads', 'V_reads', 'test_reads'))):
        raise ValueError('The complete previously fixed source cohort is required')
    with np.load(prior/'source_predictions.npz', allow_pickle=False) as z:
        ids, rx_saved, scene_saved, old = (z[k].copy() for k in ('indices', 'rx', 'scene', 'values'))
    if ids.shape != (11761,) or len(np.unique(ids)) != len(ids) or old.shape != (11761, 15):
        raise ValueError('Unique complete source predictions are required')
    cache, clean, rx, training, pv, ph = s.source_contract(args.root)
    validation, heldout = s.new_splits(cache, rx, training, pv, ph)
    received, scene = s.epoch_data(cache, 19)
    if (not np.array_equal(ids, np.r_[validation, heldout]) or np.intersect1d(ids, training).size
            or not np.array_equal(rx[ids], rx_saved) or not np.array_equal(scene[ids], scene_saved)):
        raise ValueError('The same disjoint source rows and views are required')
    pilot = args.root/'runs'/PILOT
    pilot_meta = core.read(pilot/'prediction_complete.json')
    if pilot_meta.get('status') != 'SOURCE_PREDICTIONS_FIXED' or pilot_meta.get('count') != 672:
        raise ValueError('The already fixed pilot output is required')
    with np.load(pilot/'source_predictions.npz', allow_pickle=False) as z:
        pilot_ids, pilot_values = z['indices'].copy(), z['values'].copy()
    order = np.argsort(pilot_ids); keys = pilot_ids[order]
    positions = np.searchsorted(keys, ids)
    match = (positions < len(keys)) & (keys[np.minimum(positions, len(keys)-1)] == ids)
    if int(match.sum()) != 672: raise ValueError('All fixed pilot rows must be in the full source cohort')
    physical, gate = load_source_models(args.root, 'mean')
    physical.eval().requires_grad_(False); gate.eval().requires_grad_(False)
    reader, _ = core.teacher(args.root, 'cpu'); reader.eval().requires_grad_(False)
    total = 64 if args.smoke else len(ids)
    result = []; matched = 0; tick = time.monotonic()
    for start in range(0, total, 8):
        ix = ids[start:min(total, start+8)]
        batch = compute(core.tensor(received[ix], 'cpu'), core.tensor(clean[ix], 'cpu'), physical, gate, reader, s)
        arr = np.from_dlpack(batch.cpu().contiguous())
        ref = old[start:start+len(ix)]
        if (not np.array_equal(arr[:, 0], ref[:, 5]) or not np.array_equal(arr[:, 1].astype(bool), ref[:, 3] > .8)
                or not np.array_equal(arr[:, 2], ref[:, 7])):
            raise ValueError('Raw, gate and physical predictions must match the fixed source reference')
        here = match[start:start+len(ix)]
        if here.any():
            fixed = pilot_values[order[positions[start:start+len(ix)][here]]]
            if (not np.array_equal(arr[here, 3], fixed[:, 7])
                    or not np.allclose(arr[here, 4], fixed[:, 8], rtol=1e-5, atol=1e-7)):
                raise ValueError('Previously fixed pilot decisions must replay')
            matched += int(here.sum())
        result.append(arr[:, 3:5].copy())
        if (start+len(ix)) % 512 == 0:
            print('HANKEL_FULL_SOURCE_PROGRESS', start+len(ix), 'seconds', round(time.monotonic()-tick, 1), flush=True)
    args.out.mkdir(parents=True)
    if args.smoke:
        core.write(args.out/'result.json', dict(status='PASS', records=total, source_only=True,
                   TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, test_reads=0))
        return
    if matched != 672: raise ValueError('All 672 prior fixed pilot outputs must replay')
    with (args.out/'source_predictions.npz').open('xb') as f:
        np.savez(f, indices=ids, rx=rx_saved, scene=scene_saved, prior_values=old, hankel_values=np.concatenate(result))
    core.write(args.out/'prediction_complete.json', dict(status='SOURCE_PREDICTIONS_FIXED', commit=args.commit,
               count=len(ids), pilot_replayed=matched, source_only=True, DURA_loaded=False,
               TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates=0,
               caveat='Source records previously explored; no target or ground performance claim.'))
    print('HANKEL_FULL_SOURCE_PREDICTIONS_FIXED', len(ids), flush=True)


if __name__ == '__main__': main()
