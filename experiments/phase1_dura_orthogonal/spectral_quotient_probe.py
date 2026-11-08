"""Source-only, same-record spectral descriptor diagnostic; no identity labels."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def spectral_features(iq, mode):
    """Return unit-energy complex FFT or regularized adjacent-bin ratios."""
    iq = np.asarray(iq)
    if (iq.ndim != 3 or iq.shape[1:] != (2, 256) or len(iq) < 1
            or not np.isfinite(iq).all() or mode not in ('spectrum', 'quotient')):
        raise ValueError('Finite source IQ batches of shape (N, 2, 256) required')
    z = np.fft.fft(iq[:, 0] + 1j * iq[:, 1], axis=-1, norm='ortho')
    power = np.mean(np.abs(z) ** 2, axis=1, keepdims=True)
    if np.any(power <= 1e-20):
        raise ValueError('Nonzero source IQ energy required')
    if mode == 'spectrum':
        feature = z
    else:
        # No circular edge pair: adjacent FFT indices only; no OFDM assumptions.
        left, right = z[:, :-1], z[:, 1:]
        feature = right * left.conj() / (np.abs(left) ** 2 + .01 * power)
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-20):
        raise ValueError('Finite nonzero descriptor required')
    return feature / norm


def match_ranks(clean_features, observed_features):
    """Pessimistic same-record retrieval ranks; larger is worse."""
    a, b = np.asarray(clean_features), np.asarray(observed_features)
    if (a.ndim != 2 or a.shape != b.shape or len(a) < 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Matching finite nonempty feature matrices required')
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(an <= 1e-20) or np.any(bn <= 1e-20):
        raise ValueError('Nonzero descriptors required')
    similarity = np.abs((b / bn) @ (a / an).conj().T)
    matched = np.diag(similarity)
    return np.sum(similarity >= matched[:, None], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    from experiments.pamf_phase1_flow import experiment as core
    ids, rx, scene, split, received, clean = source_inputs(args.root)
    # Source-only smoke is part of the job, before the full diagnostic.
    for mode in ('spectrum', 'quotient'):
        smoke = spectral_features(np.asarray(clean[ids[:8]]), mode)
        ranks = match_ranks(smoke, smoke)
        if ranks.shape != (8,) or np.any((ranks < 1) | (ranks > 8)):
            raise ValueError('Real-source descriptor smoke failed')
    print('SPECTRAL_QUOTIENT_SOURCE_SMOKE_PASS 8', flush=True)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            indices = ids[np.isin(rx, receivers) & (scene == view)]
            if len(indices) != 48:
                raise ValueError('Each source receiver group and view must contain 48 records')
            original = np.asarray(clean[indices])
            damaged = np.asarray(received[indices])
            metrics = {}
            for mode in ('spectrum', 'quotient'):
                ranks = match_ranks(spectral_features(original, mode), spectral_features(damaged, mode))
                metrics[mode] = {'top1': float(np.mean(ranks == 1)),
                                 'mean_reciprocal_rank': float(np.mean(1.0 / ranks)),
                                 'median_rank': float(np.median(ranks))}
            groups[group + '/' + view] = {'count': len(indices), 'metrics': metrics}
    result = {'status': 'SOURCE_DESCRIPTOR_DIAGNOSTIC_COMPLETE', 'commit': args.commit,
              'records': len(ids), 'groups': groups, 'regularization': .01,
              'source_only': True, 'TX_labels_read': 0, 'U_s_truth_reads': 0,
              'F_weights_read': 0, 'F_outputs_read': 0, 'V_reads': 0,
              'target_IQ_reads': 0, 'target_truth_reads': 0, 'model_updates': 0,
              'elapsed_seconds': time.monotonic() - started,
              'caveat': 'Clean source IQ is used as reference. Same-record retrieval is not transmitter identity accuracy, a deployable receiver, or evidence of target improvement.'}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('SPECTRAL_QUOTIENT_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
