"""Select temporal positives and negatives in native PAMF source batches."""

import numpy as np


MAX_GAP_NS = 5_000_000_000


def prototype_positive_slots(timestamps=None):
    """Select two other members of each batch's four-packet beam groups."""
    positions = np.arange(4)
    if timestamps is None:
        # SourceBatches permutes each beam before forming the groups.
        local = (positions[:, None] + np.array([1, 2])) % 4
    else:
        grouped = timestamps.reshape(8, 4)
        gaps = np.abs(grouped[:, :, None] - grouped[:, None, :])
        gaps[:, positions, positions] = -1
        local = np.argsort(gaps, axis=-1, kind="stable")[..., -2:]
    return (local + (4 * np.arange(8))[:, None, None]).reshape(32, 2)


def _assign_nearest_other_group(neighbors, rows, timestamps, group_ids):
    order = np.argsort(timestamps[rows], kind="stable")
    rows = rows[order]
    times = timestamps[rows]
    groups = group_ids[order]
    cuts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(rows)]
    # The nearest different group is immediately outside the contiguous run.
    first = np.repeat(cuts[:-1], np.diff(cuts))
    stop = np.repeat(cuts[1:], np.diff(cuts))
    left = (first - 1).clip(min=0)
    right = stop.clip(max=len(rows) - 1)
    earlier = np.where(first > 0, times - times[left], np.inf)
    later = np.where(stop < len(rows), times[right] - times, np.inf)
    feasible = np.minimum(earlier, later) <= MAX_GAP_NS
    nearest = np.where(earlier <= later, left, right)
    neighbors[rows[feasible]] = rows[nearest[feasible]]


def nearest_temporal_negatives(timestamps, labels, satellites, *, cross_satellite=False):
    """Map each labeled L row to a nearby other beam or another satellite."""
    known = np.flatnonzero(labels >= 0)
    neighbors = np.full(len(labels), -1, dtype=np.int64)
    if not len(known):
        return neighbors
    satellite_ids = satellites[labels[known]]
    if cross_satellite:
        _assign_nearest_other_group(neighbors, known, timestamps, satellite_ids)
    else:
        for satellite in np.unique(satellite_ids):
            rows = known[satellite_ids == satellite]
            _assign_nearest_other_group(neighbors, rows, timestamps, labels[rows])
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
