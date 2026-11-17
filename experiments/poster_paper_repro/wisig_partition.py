"""Fixed, disclosed cross-day WiSig diagnostic; not the author's dataset."""
import numpy as np


def select_indices(source_counts, target_counts, *, seed):
    eligible = sorted(label for label in source_counts.keys() & target_counts.keys()
                      if source_counts[label] >= 240 and target_counts[label] >= 110)
    if len(eligible) < 5:
        raise ValueError('five sufficiently sampled transmitters are required')
    rng = np.random.default_rng(seed)
    source, target, truth = [], [], []
    for label in eligible[:5]:
        source_order = rng.permutation(source_counts[label]).tolist()
        target_order = rng.permutation(target_counts[label]).tolist()
        source.append({'id': label, 'train': source_order[:200], 'validation': source_order[200:240]})
        target.append({'id': label, 'support': target_order[:10]})
        truth.append({'id': label, 'query': target_order[10:110]})
    return source, target, truth
