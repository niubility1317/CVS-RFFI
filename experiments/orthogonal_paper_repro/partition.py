"""Construct a disclosed WiSig partition using the paper's published counts.

The paper does not specify the receiver, transmitter order or split seed. The
caller must fix these before training; this is not the authors' exact split.
"""
from __future__ import annotations

import numpy as np


def build_partition(counts: dict[str, int], *, seed: int) -> tuple[dict, dict]:
    """Return separate training and truth-side record indices; no IQ is read."""
    eligible = sorted(key for key, value in counts.items() if value >= 50)
    if len(eligible) < 130:
        raise ValueError('a single receiver requires at least 130 classes with 50 samples')
    # Choose the lexicographically first 130 eligible transmitters, not by score.
    chosen = eligible[:130]
    rng = np.random.default_rng(seed)
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    base, increments, evaluation = [], [[] for _ in range(7)], {}
    for position, class_id in enumerate(chosen):
        indices = rng.permutation(int(counts[class_id])).tolist()
        train_count = int(.8 * counts[class_id]) if position < 60 else 5
        row = {'id': class_id, 'indices': indices[:train_count]}
        evaluation[class_id] = indices[train_count:]
        if position < 60:
            base.append(row)
        else:
            increments[(position - 60) // 10].append(row)
    return ({'base': base, 'increments': increments, 'seed': int(seed)},
            {'classes': evaluation, 'seed': int(seed)})
