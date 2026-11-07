"""Count a class-collision lower bound without reading source data or TX labels."""
from collections import Counter
import json

import numpy as np
import torch


def minimum_same_class_ordered_pairs(unique_records, classes):
    """Pigeonhole lower bound among distinct records in a fixed number of classes."""
    if not isinstance(unique_records, int) or not isinstance(classes, int) or unique_records < 0 or classes < 1:
        raise ValueError('Nonnegative record count and positive class count required')
    q, r = divmod(unique_records, classes)
    return r*q*(q+1) + (classes-r)*q*(q-1)


def sampler_bound():
    # Fixed preregistered source sampler, not data-dependent and not model training.
    fit, views, classes, batch, epochs = 32768, 3, 6, 32, 5
    generator = torch.Generator().manual_seed(2026110613)
    histogram = Counter()
    negative_pairs = minimum_collisions = 0
    for _ in range(epochs):
        order = torch.randperm(views*fit, generator=generator).numpy()
        for start in range(0, len(order), batch):
            ids = order[start:start+batch] % fit
            if len(ids) != batch:
                raise ValueError('Fixed source budget must contain complete batches')
            _, counts = np.unique(ids, return_counts=True)
            unique = len(counts)
            histogram[unique] += 1
            negative_pairs += batch**2-int((counts**2).sum())
            minimum_collisions += minimum_same_class_ordered_pairs(unique, classes)
    return dict(batches=sum(histogram.values()), unique_id_histogram=dict(sorted(histogram.items())),
        same_tx_negative_pairs_lower_bound=minimum_collisions, record_negative_pairs=negative_pairs,
        aggregate_lower_bound_fraction=minimum_collisions/negative_pairs,
        source_IQ_reads=0, TX_labels_read=0, F_weights_read=0, V_reads=0, target_reads=0,
        caveat='Counts depend on the fixed sampler and six-class assumption; same-device negatives are intentional for instance discrimination, not verified representation damage.')


if __name__ == '__main__':
    print(json.dumps(sampler_bound(), indent=2, allow_nan=False))
