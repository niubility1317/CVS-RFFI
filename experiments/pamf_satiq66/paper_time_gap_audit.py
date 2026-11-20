"""Measure fixed source beam accuracy by distance from same-beam training captures."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import encode
from .paper_quality_audit import matched_beam_contrast
from .paper_readout import load_source_model
from .paper_train import source_inputs
from .train import device_array


GAP_EDGES_SECONDS = (5, 30, 300)


def nearest_seconds(training_time, training_labels, query_time, query_labels):
    order = np.argsort(training_labels, kind="stable")
    labels = training_labels[order]
    gap = np.empty(len(query_time), dtype=np.float64)
    for label in np.unique(query_labels):
        start, stop = np.searchsorted(labels, [label, label + 1])
        if start == stop:
            raise ValueError(f"Expected training timestamps for beam {label}")
        times = np.sort(training_time[order[start:stop]])
        selected = query_labels == label
        query = query_time[selected]
        right = np.searchsorted(times, query)
        left = right - 1
        earlier = np.where(left >= 0, query - times[left.clip(0)], np.inf)
        later = np.where(right < len(times), times[right.clip(max=len(times) - 1)] - query, np.inf)
        gap[selected] = np.minimum(earlier, later) / 1_000_000_000
    return gap


@torch.no_grad()
def audit(source, cache, checkpoint, seed, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    device = torch.device("cpu")
    model, cfg, updates = load_source_model(source, checkpoint, classes, device)
    if (updates != 846750 or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
            or cfg["input_samples"] != 11000 or cfg["source_roles"] != ["L", "V"]
            or Path(cfg["cache"]).resolve() != cache):
        raise ValueError("Expected the fixed matched PAMF E30 source checkpoint")
    role = "L" if smoke else "V"
    raw, labels, _ = roles[role]
    known = np.flatnonzero(labels >= 0)
    if smoke:
        known = known[:32]
    targets = labels[known]
    embeddings, = encode(model, cfg["arm"], raw, known, device)
    weight = F.normalize(model.head.weight, dim=1)
    pred, ce = [], []
    for start in range(0, len(targets), 256):
        target = device_array(targets[start:start + 256].astype(np.int64), device)
        scores = 30 * F.linear(embeddings[start:start + 256], weight)
        pred.append(scores.argmax(1))
        ce.append(F.cross_entropy(scores, target, reduction="none"))
    prediction = np.from_dlpack(torch.cat(pred))
    losses = np.from_dlpack(torch.cat(ce))

    # All predictions are fixed before timestamps are read from source metadata.
    with np.load(source / "L_index.npz", allow_pickle=False) as f:
        train_ticks = f["timestamp_ns"]
    if smoke:
        training = np.ones(len(train_ticks), dtype=bool)
        training[known] = False
        gap = nearest_seconds(train_ticks[training], labels[training], train_ticks[known], targets)
    else:
        with np.load(source / "V_index.npz", allow_pickle=False) as f:
            ticks = f["timestamp_ns"][known]
        gap = nearest_seconds(train_ticks, roles["L"][1], ticks, targets)
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite CE in the frozen source readout")
    strata = np.searchsorted(GAP_EDGES_SECONDS, gap, side="left")
    correct = prediction == targets
    satellite = classes[:, 0]
    sat_ok = satellite[prediction] == satellite[targets]

    def describe(mask):
        return dict(records=int(mask.sum()), hits=int(correct[mask].sum()), satellite_hits=int(sat_ok[mask].sum()),
                    beams=int(np.unique(targets[mask]).size),
                    ce=float(losses[mask].mean(dtype=np.float64)) if mask.any() else None)

    extremes = np.where(strata == 0, 0, np.where(strata >= 2, 3, 1))
    return dict(checkpoint=str(checkpoint), updates=updates, candidates=len(classes), smoke=smoke,
                edges_seconds=GAP_EDGES_SECONDS, seed=seed,
                all=describe(np.ones(len(targets), dtype=bool)),
                strata=[describe(strata == k) for k in range(4)],
                far_minus_near_within_beam=matched_beam_contrast(targets, extremes, correct, seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for attr in ("source", "cache", "checkpoint", "output"):
        parser.add_argument(attr, type=Path)
    parser.add_argument("--seed", type=int, default=2026111923)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.source.resolve(), args.cache.resolve(), args.checkpoint.resolve(), args.seed, args.smoke)
    write_json(args.output, result)
