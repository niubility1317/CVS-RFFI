"""Recompute a class-collision lower bound for the fixed source rank sampler.

Uses no data, labels, model, or target output. Duplicate record views are
positives; representative records give a conservative collision bound.
"""
import collections
import json
import numpy as np


def run():
    fit, views, batch, epochs, classes, seed = 32768, 3, 32, 20, 6, 2026110902
    rng = np.random.default_rng(seed)
    histogram = collections.Counter()
    total = lower = 0
    for _ in range(epochs):
        ids = rng.permutation(views * fit).reshape(-1, batch) % fit
        ordered = np.sort(ids, axis=1)
        unique = 1 + np.count_nonzero(np.diff(ordered, axis=1), axis=1)
        histogram.update(map(int, unique))
        same_record = ids[:, :, None] == ids[:, None, :]
        total += int(np.sum(batch * batch - same_record.sum(axis=(1, 2))))
        q, r = unique // classes, unique % classes
        lower += int(np.sum(r * q * (q + 1) + (classes - r) * q * (q - 1)))
    return dict(epochs=epochs, batch=batch, batches=sum(histogram.values()), seed=seed,
                unique_records_histogram=dict(sorted(histogram.items())),
                different_record_ordered_pairs=total,
                same_device_ordered_pairs_lower_bound=lower,
                lower_bound_fraction=lower / total, IQ_reads=0, TX_labels_read=0,
                F_reads=0, target_reads=0)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
