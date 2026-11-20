"""Place near-time same-satellite negatives in native PAMF source batches."""

import numpy as np


MAX_GAP_NS = 5_000_000_000


def nearest_temporal_negatives(timestamps, labels, satellites):
    """Map each labeled L row to its closest other beam on the same satellite."""
    known = np.flatnonzero(labels >= 0)
    satellite_ids = satellites[labels[known]]
    neighbors = np.full(len(labels), -1, dtype=np.int64)
    for satellite in np.unique(satellite_ids):
        rows = known[satellite_ids == satellite]
        rows = rows[np.argsort(timestamps[rows], kind="stable")]
        times = timestamps[rows]
        beams = labels[rows]
        cuts = np.r_[0, np.flatnonzero(beams[1:] != beams[:-1]) + 1, len(rows)]
        # The nearest other beam must be immediately outside the contiguous run.
        first = np.repeat(cuts[:-1], np.diff(cuts))
        stop = np.repeat(cuts[1:], np.diff(cuts))
        left = (first - 1).clip(min=0)
        right = stop.clip(max=len(rows) - 1)
        earlier = np.where(first > 0, times - times[left], np.inf)
        later = np.where(stop < len(rows), times[right] - times, np.inf)
        feasible = np.minimum(earlier, later) <= MAX_GAP_NS
        nearest = np.where(earlier <= later, left, right)
        neighbors[rows[feasible]] = rows[nearest[feasible]]
    return neighbors


def pair_temporal_negatives(batches, labels, neighbors):
    """Swap only complete beam groups and keep each batch's eight beams distinct."""
    paired = batches.copy().reshape(-1, 8, 4)
    position = np.full(len(labels), -1, dtype=np.int64)
    position[paired.ravel()] = np.repeat(np.arange(len(paired) * 8), 4)
    beams = labels[paired[:, :, 0]].copy()
    for batch in range(len(paired)):
        for anchor in paired[batch, :-1].ravel():
            neighbor = neighbors[anchor]
            if neighbor < 0 or position[neighbor] < 0:
                continue
            donor, group = divmod(position[neighbor], 8)
            if donor <= batch or beams[donor, group] in beams[batch, :-1]:
                continue
            if beams[batch, -1] in np.delete(beams[donor], group):
                continue
            outgoing = paired[batch, -1].copy()
            paired[batch, -1] = paired[donor, group]
            paired[donor, group] = outgoing
            beams[batch, -1], beams[donor, group] = beams[donor, group], beams[batch, -1]
            position[paired[batch, -1]] = batch * 8 + 7
            position[outgoing] = donor * 8 + group
            break
    return paired.reshape(batches.shape)
