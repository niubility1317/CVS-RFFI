"""Prepare immutable ADS-B train/query files using the CSIL feature recipe.

The MoPC paper's exact ADS-B preprocessing and class IDs are not public in
its repository. This is an explicitly labelled, independently reproducible
proxy, not the authors' original split or a claim of exact reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def split_rows(ids, seed=1731, max_records=1000, train_limit=200):
    ids = np.asarray(ids).reshape(-1)
    labels, counts = np.unique(ids, return_counts=True)
    # Deterministic tie break by physical ID, not performance.
    ranked = np.lexsort((labels, -counts))[:100]
    if len(ranked) != 100 or np.min(counts[ranked]) < 334:
        raise ValueError('Need 100 classes with enough disjoint train/query records')
    rng = np.random.default_rng(seed)
    ranked = rng.permutation(ranked)
    tr_i, tr_y, qu_i, qu_y = [], [], [], []
    for label, index in enumerate(ranked):
        rows = rng.permutation(np.flatnonzero(ids == labels[index]))[:max_records]
        cut = int(len(rows) * 0.6)
        if cut < train_limit or cut == len(rows):
            raise ValueError('Insufficient training or query records')
        tr_i.extend(rows[:train_limit]); tr_y.extend([label] * train_limit)
        qu_i.extend(rows[cut:]); qu_y.extend([label] * (len(rows) - cut))
    return (np.asarray(tr_i), np.asarray(tr_y), np.asarray(qu_i),
            np.asarray(qu_y), labels[ranked], counts[ranked])


def csil_features(iq):
    """CSIL extractNoise.m + makeDataTensor.m; rows are 1024 complex samples."""
    x = np.asarray(iq, dtype=np.complex128).copy()
    if x.ndim != 2 or x.shape[1] != 1024:
        raise ValueError('Expected complex records [N,1024]')
    x[:, :256] = 0  # CSIL script removes the ICAO-bearing prefix.
    amp = np.abs(x)
    mean = amp.mean(axis=1, keepdims=True)
    std = amp.std(axis=1, ddof=1, keepdims=True)
    clean = amp * (np.abs(amp - mean) <= 3 * std)
    lo, hi = clean.min(axis=1, keepdims=True), clean.max(axis=1, keepdims=True)
    for _ in range(3):
        high = np.abs(clean - hi) <= np.abs(clean - lo)
        nh = high.sum(axis=1, keepdims=True)
        nl = (~high).sum(axis=1, keepdims=True)
        if np.any(nh == 0) or np.any(nl == 0):
            raise ValueError('Degenerate CSIL medoid cluster')
        hi = (clean * high).sum(axis=1, keepdims=True) / nh
        lo = (clean * ~high).sum(axis=1, keepdims=True) / nl
    medoid = np.where(clean >= (hi + lo) / 2, hi, lo)
    noise = medoid - clean
    residual = np.fft.fftshift(np.fft.fft(x, axis=1) - np.fft.fft(medoid, axis=1), axes=1)
    channels = np.stack((noise, residual.real, residual.imag), axis=1)
    # MATLAB reshape(...,[32,32,3,N]) is column-major within each channel.
    channels = channels.reshape(-1, 3, 32, 32).transpose(0, 1, 3, 2)
    low = channels.min(axis=(1, 2, 3), keepdims=True)
    span = channels.max(axis=(1, 2, 3), keepdims=True) - low
    if np.any(span == 0) or not np.isfinite(channels).all():
        raise ValueError('Non-finite or constant example')
    return ((channels - low) / span).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args()
    raw = loadmat(args.raw, variable_names=['icaoLst', 'rawIMatrix', 'rawQMatrix'])
    ids = raw['icaoLst'].reshape(-1)
    ti, ty, qi, qy, selected, counts = split_rows(ids)
    ri, rq = raw['rawIMatrix'].reshape(len(ids), -1), raw['rawQMatrix'].reshape(len(ids), -1)
    if ri.shape != rq.shape or ri.shape[1] < 1024:
        raise ValueError('Unexpected original IQ dimensions')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    for name, indices, truth in [('train', ti, ty), ('query', qi, qy)]:
        arr = np.lib.format.open_memmap(out / (name + '_x.npy'), mode='w+', dtype='float32',
                                       shape=(len(indices), 3, 32, 32))
        for pos in range(0, len(indices), 128):
            subset = indices[pos:pos + 128]
            iq = ri[subset, :1024] + 1j * rq[subset, :1024]
            arr[pos:pos + len(subset)] = csil_features(iq)
        arr.flush(); del arr
        np.save(out / (name + '_labels.npy'), truth)
        np.save(out / (name + '_rows.npy'), indices)
        print(name, len(indices), flush=True)
    (out / 'data_manifest.json').write_text(json.dumps({
        'kind': 'CSIL preprocessing proxy; not verified MoPC original preprocessing',
        'raw_name': Path(args.raw).name, 'num_raw_records': len(ids),
        'num_raw_classes': len(np.unique(ids)), 'classes_above_500': int((np.unique(ids, return_counts=True)[1] > 500).sum()),
        'selected_raw_ids': selected.tolist(), 'selected_counts': counts.tolist(),
        'split_seed': 1731, 'max_records_per_class': 1000,
        'split_train_ratio': 0.6, 'train_limit_per_class': 200,
        'stage_classes': [50, 25, 25], 'query_records': len(qi),
        'query_prefix_lengths': [int((qy < n).sum()) for n in (50, 75, 100)],
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
