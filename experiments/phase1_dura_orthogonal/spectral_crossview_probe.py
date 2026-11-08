"""Source-only retrieval across separately cached impaired IQ views."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.phase1_dura_orthogonal.spectral_quotient_probe import spectral_features, match_ranks

EPOCHS = ('21', '22', '23')
PAIRS = (('21', '22'), ('21', '23'), ('22', '23'))


def summarize_ranks(ranks, candidates):
    ranks = np.asarray(ranks)
    if (ranks.ndim != 1 or len(ranks) == 0 or candidates < 2
            or not np.issubdtype(ranks.dtype, np.integer)
            or np.any((ranks < 1) | (ranks > candidates))):
        raise ValueError('Valid nonempty integer retrieval ranks required')
    return {'count': len(ranks), 'top1': float(np.mean(ranks == 1)),
            'mean_reciprocal_rank': float(np.mean(1.0 / ranks)),
            'median_rank': float(np.median(ranks))}


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
    ids, rx, _, _, _, clean = source_inputs(args.root)
    fresh = args.root / 'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    original = args.root / 'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache'
    if (fresh / 'train/physical_ids.npy').resolve() != (original / 'train/physical_ids.npy').resolve():
        raise ValueError('Source views must reference original physical IDs')
    views, scenes = {}, {}
    for epoch in EPOCHS:
        folder = fresh / 'epochs' / epoch
        values = np.load(folder / 'received.npy', mmap_mode='r', allow_pickle=False)
        with np.load(folder / 'groups.npz', allow_pickle=False) as metadata:
            scene = metadata['scene'].copy()
        if values.shape != clean.shape or values.dtype != np.float32 or scene.shape != (len(clean),):
            raise ValueError('Aligned source-view arrays required')
        views[epoch], scenes[epoch] = values, scene
    # Real-source no-query smoke precedes all group comparisons.
    for mode in ('spectrum', 'quotient'):
        a = spectral_features(np.asarray(views['21'][ids[:8]]), mode)
        b = spectral_features(np.asarray(views['22'][ids[:8]]), mode)
        summarize_ranks(match_ranks(a, b), 8)
    print('SPECTRAL_CROSSVIEW_SOURCE_SMOKE_PASS 8', flush=True)
    result_groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        selected = ids[np.isin(rx, receivers)]
        if len(selected) != 336:
            raise ValueError('Expected 336 source records per receiver group')
        for first, second in PAIRS:
            a = np.asarray(views[first][selected])
            b = np.asarray(views[second][selected])
            query_scenes = scenes[second][selected]
            if not np.isin(query_scenes, core.VIEWS).all():
                raise ValueError('Only registered source views permitted')
            metrics = {}
            for mode in ('spectrum', 'quotient'):
                ranks = match_ranks(spectral_features(a, mode), spectral_features(b, mode))
                by_scene = {}
                for view in core.VIEWS:
                    subset = ranks[query_scenes == view]
                    if len(subset) == 0:
                        raise ValueError('Every query scene requires source records')
                    by_scene[view] = summarize_ranks(subset, len(selected))
                metrics[mode] = {'overall': summarize_ranks(ranks, len(selected)), 'by_query_scene': by_scene}
            result_groups[group + '/' + first + '-' + second] = {'count': len(selected), 'metrics': metrics}
    result = {'status': 'SOURCE_CROSSVIEW_DIAGNOSTIC_COMPLETE', 'commit': args.commit,
              'records': len(ids), 'groups': result_groups, 'regularization': .01,
              'source_epochs': list(EPOCHS), 'source_only': True, 'TX_labels_read': 0,
              'U_s_truth_reads': 0, 'F_weights_read': 0, 'F_outputs_read': 0, 'V_reads': 0,
              'target_IQ_reads': 0, 'target_truth_reads': 0, 'source_clean_IQ_values_read': 0,
              'model_updates': 0, 'elapsed_seconds': time.monotonic() - started,
              'caveat': 'Ranks use the known correspondence of source record IDs for analysis; same-record retrieval is not transmitter identity accuracy or target performance.'}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('SPECTRAL_CROSSVIEW_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
