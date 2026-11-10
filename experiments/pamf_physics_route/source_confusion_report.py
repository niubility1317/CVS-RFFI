"""Summarize previously frozen source-V predictions by receiver, day and class."""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path):
    with np.load(path, allow_pickle=False) as data:
        required = {'logits', 'y', 'ids', 'rx', 'day', 'views', 'seeds'}
        if not required.issubset(data.files):
            raise ValueError('Saved source-V logits and physical metadata required')
        logits, y, ids = data['logits'], data['y'], data['ids']
        rx, day, views, seeds = data['rx'], data['day'], data['views'], data['seeds']
        n, classes = len(y), 6
        if (n != 27000 or len(np.unique(ids)) != n or rx.shape != y.shape or day.shape != y.shape
                or logits.shape != (3, 8, n, classes) or len(views) != 8
                or list(seeds) != [2026092701, 2026092702, 2026092703]
                or np.any((y < 0) | (y >= classes)) or not np.isfinite(logits).all()):
            raise ValueError('Complete finite three-seed six-class source V required')
        records = []
        for i, seed in enumerate(seeds):
            for j, view in enumerate(views):
                scores = logits[i, j]
                predicted = scores.argmax(1)
                shifted = scores - scores.max(1, keepdims=True)
                probabilities = np.exp(shifted)
                confidence = probabilities.max(1) / probabilities.sum(1)
                accepted = confidence >= .95
                groups = [('overall', 'all', np.ones(n, dtype=bool))]
                groups += [('receiver', str(v), rx == v) for v in np.unique(rx)]
                groups += [('day', str(v), day == v) for v in np.unique(day)]
                for dimension, stratum, mask in groups:
                    confusion = np.bincount(y[mask] * classes + predicted[mask],
                                            minlength=classes ** 2).reshape(classes, classes)
                    records.append(dict(seed=int(seed), view=str(view), dimension=dimension, stratum=stratum,
                        count=int(mask.sum()), confusion=confusion.tolist(),
                        accepted=int((mask & accepted).sum()),
                        accepted_wrong=int((mask & accepted & (predicted != y)).sum())))
    return dict(status='SOURCE_V_CONFUSION_COMPLETE', role='V', source_path=str(Path(path)),
                training_reads=0, target_reads=0, U_reads=0, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.predictions), allow_nan=False))
