"""Source-only parity and timing probe; never modifies the ground predictor."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import SOURCE_PILOT, source_inputs
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.train_source import extract_context


@torch.no_grad()
def run_batch(x, model, dura, reader, include_anchor):
    anchor, direction, variance = extract_context(dura, x)
    restored = model(x, anchor, direction, variance)
    raw = reader.classify_features(reader.features(x))
    anchor_logits = reader.classify_features(reader.features(anchor)) if include_anchor else None
    candidate = reader.classify_features(reader.features(restored))
    if (not torch.isfinite(raw).all() or not torch.isfinite(candidate).all()
            or (anchor_logits is not None and not torch.isfinite(anchor_logits).all())):
        raise FloatingPointError('Finite source logits required')
    return raw, anchor_logits, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    selected = []
    for receivers in ((4, 6), (8,)):
        for view in core.VIEWS:
            match = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(match) < 8:
                raise ValueError('Eight matching source records per group required')
            selected.extend(match[:8].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    with np.load(args.root/'runs'/SOURCE_PILOT/'source_predictions.npz', allow_pickle=False) as z:
        expected = z['values'][selected][:, [3, 5]].astype(np.int64)
    model, dura, reader = load_fixed(args.root, args.source, args.candidate_commit)
    durations = {'standard': 0., 'skip_anchor_reader': 0.}
    max_diff = {'raw': 0., 'restored': 0.}
    mismatch = {'raw': 0, 'restored': 0, 'pilot_raw_anchor': 0}
    for start in range(0, len(selected), 8):
        x = core.tensor(received[ids[selected[start:start+8]]], 'cpu')
        results = {}
        # Alternate measurement order to reduce simple warmup/order bias.
        for key in (('standard', 'skip_anchor_reader') if start % 16 == 0 else ('skip_anchor_reader', 'standard')):
            t = time.perf_counter()
            results[key] = run_batch(x, model, dura, reader, key == 'standard')
            durations[key] += time.perf_counter() - t
        standard, fast = results['standard'], results['skip_anchor_reader']
        for name, i in (('raw', 0), ('restored', 2)):
            diff = (standard[i]-fast[i]).abs().max().item()
            max_diff[name] = max(max_diff[name], diff)
            mismatch[name] += int((standard[i].argmax(1) != fast[i].argmax(1)).sum())
        actual = np.from_dlpack(torch.stack((standard[0].argmax(1), standard[1].argmax(1)), 1).cpu().contiguous())
        mismatch['pilot_raw_anchor'] += int(np.count_nonzero(actual != expected[start:start+len(x)]))
    if any(mismatch.values()) or any(max_diff.values()):
        raise ValueError('Source logits or fixed pilot predictions differ: '+str((mismatch, max_diff)))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_PARITY_PASS', commit=args.commit,
        candidate=args.source, candidate_commit=args.candidate_commit, count=len(selected), batch_size=8,
        threads=2, durations_seconds=durations, max_logit_abs_diff=max_diff, decision_mismatches=mismatch,
        TX_truth_reads=0, U_s_truth_reads=0, V_reads=0, target_reads=0,
        parity_scope='112 source records only; target prediction equality not established'))
    print('SOURCE_READER_SKIP_PROBE_PASS', len(selected), durations, flush=True)


if __name__ == '__main__':
    main()
