"""Frozen Phase1 source-only agreement with a clean-signal reader proxy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs, SOURCE_PILOT
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.train_source import extract_context

CANDIDATES = {
    'fixed_views': ('20261108-phase1-direct-large-timefreq-stable-source-s2026092701-r01', '4bc1aa7209c1ad7ac305f2802ebc6dee195c207a'),
    'source_physics': ('20261108-phase1-direct-timefreq-source-physics-s2026092701-r01', '7806bb102cd190c9c885706a7a71ac49ce453cb5'),
    'rotating_views': ('20261108-phase1-direct-timefreq-diverse-source-s2026092701-r01', '0e9d62254690f700eef7111c06fb7db044155998'),
}


def summarize(proxy, predictions, rx, scene, *, views, per_group=48):
    proxy = np.asarray(proxy)
    rx = np.asarray(rx); scene = np.asarray(scene)
    if proxy.ndim != 1 or rx.shape != proxy.shape or scene.shape != proxy.shape:
        raise ValueError('Matching source metadata required')
    if np.any((proxy < 0) | (proxy > 5)) or 'dura' not in predictions:
        raise ValueError('Valid six-class source proxy and DURA required')
    for name, value in predictions.items():
        if np.asarray(value).shape != proxy.shape or np.any((value < 0) | (value > 5)):
            raise ValueError('Valid predicted class arrays required for ' + name)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in views:
            mask = np.isin(rx, receivers) & (scene == view)
            if int(mask.sum()) != per_group:
                raise ValueError('Fixed source group size required')
            matches = {name: int(np.count_nonzero(np.asarray(value)[mask] == proxy[mask]))
                       for name, value in predictions.items()}
            groups[group + '/' + view] = dict(count=per_group, matches=matches,
                delta_vs_dura={name: count - matches['dura'] for name, count in matches.items()})
    return groups


@torch.no_grad()
def classes(reader, x):
    logits = reader.classify_features(reader.features(x))
    if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
        raise ValueError('Finite frozen reader outputs required')
    return logits.argmax(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    with np.load(args.root/'runs'/SOURCE_PILOT/'source_predictions.npz', allow_pickle=False) as z:
        if not np.array_equal(z['indices'], ids):
            raise ValueError('Matching fixed source pilot IDs required')
        saved = z['values'][:, [1, 3, 5]].astype(np.int64)
    models = {}
    dura = reader = None
    for name, (source, commit) in CANDIDATES.items():
        state = torch.load(args.root/'runs'/source/'final.pt', map_location='cpu', weights_only=False)
        if np.intersect1d(ids, np.asarray(state['fit_ids'])).size:
            raise ValueError('Source pilot must be disjoint from candidate fit')
        model, anchor, classifier = load_fixed(args.root, source, commit)
        models[name] = model
        if dura is None:
            dura, reader = anchor, classifier
    args.out.mkdir(parents=True)
    batches = []
    with torch.no_grad():
        for start in range(0, len(ids), 8):
            selected = ids[start:start+8]
            x = core.tensor(received[selected], 'cpu')
            y = core.tensor(clean[selected], 'cpu')
            anchor, _, _ = extract_context(dura, x)
            columns = [classes(reader, y), classes(reader, x), classes(reader, anchor)]
            for model in models.values():
                columns.append(classes(reader, model(x, validate=False)))
            block = torch.stack(columns, 1).cpu().numpy()
            if not np.array_equal(block[:, :3], saved[start:start+len(x)]):
                raise ValueError('Frozen clean, raw or DURA source proxy mismatch')
            batches.append(block)
            if start == 0:
                print('FROZEN_SOURCE_PROXY_SMOKE_PASS', len(x), flush=True)
            if (start+len(x)) % 168 == 0:
                print('FROZEN_SOURCE_PROXY_PROGRESS', start+len(x), flush=True)
    values = np.concatenate(batches, 0)
    names = ['raw', 'dura', *models.keys()]
    predictions = {name: values[:, i+1] for i, name in enumerate(names)}
    with (args.out/'source_predictions.npz').open('xb') as f:
        np.savez(f, ids=ids, rx=rx, scene=scene, clean_proxy=values[:, 0], **predictions)
    result = dict(status='SOURCE_FROZEN_PROXY_COMPLETE', commit=args.commit, records=len(ids),
        candidate_commits={name: commit for name, (_, commit) in CANDIDATES.items()},
        metric='agreement_with_frozen_clean_pseudoclass_not_accuracy',
        groups=summarize(values[:, 0], predictions, rx, scene, views=core.VIEWS),
        TX_labels_read=0, U_s_truth_reads=0, V_reads=0, target_reads=0, model_updates=0,
        note='Previously explored source data; frozen clean prediction is a proxy, not TX truth.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print('FROZEN_SOURCE_PROXY_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
