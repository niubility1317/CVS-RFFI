"""Compare one or more source prototypes for each of the existing PAMF classes."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import encode
from .paper_covariance_head import summarize
from .paper_direct_branches import blend_logits
from .paper_grid_paired_audit import paired_delta_interval
from .paper_readout import load_source_model
from .paper_train import source_inputs
from .train import device_array


ALPHAS = (0., .10, .20, .35, .50)
SLOTS = 4


def split_enrollment(labels, class_count, seed, fit_limit=32, calibration_limit=8):
    rng = np.random.default_rng(seed)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(class_count + 1))
    fit, calibration = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        n = hi - lo
        if n < 4:
            raise ValueError("Every eligible beam needs four source packets")
        hold = min(calibration_limit, max(1, n // 4))
        take = min(fit_limit, n - hold)
        chosen = rng.choice(order[lo:hi], take + hold, replace=False)
        fit.append(chosen[:take])
        calibration.append(chosen[take:])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(calibration))


def fit_centers(features, labels, ticks, class_count):
    order = np.lexsort((ticks, labels))
    bounds = np.searchsorted(labels[order], np.arange(class_count + 1))
    ordered = features[device_array(order.astype(np.int64), features.device)]
    single = features.new_zeros(class_count, features.shape[1])
    multi = features.new_zeros(class_count, SLOTS, features.shape[1])
    log_mass = features.new_full((class_count, SLOTS), -torch.inf)
    for label, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        group = ordered[lo:hi]
        if len(group) < 2:
            raise ValueError("Each prototype needs at least two enrolled source packets")
        single[label] = group.mean(0)
        count = min(SLOTS, len(group) // 2)
        for i, block in enumerate(torch.tensor_split(group, count)):
            multi[label, i] = block.mean(0)
            log_mass[label, i] = len(block) / len(group)
    valid = log_mass > 0
    log_mass[valid] = log_mass[valid].log()
    return F.normalize(single, dim=1), F.normalize(multi, dim=2), log_mass


def prototype_logits(x, centroids, log_mass):
    sub_scores = (30 * F.linear(x, centroids.flatten(0, 1))).reshape(len(x), *log_mass.shape)
    return torch.logsumexp(sub_scores + log_mass, dim=-1)


def score_batches(features, targets, weight, single, multi, mass):
    for x, y in zip(features.split(128), targets.split(128), strict=True):
        yield 30 * F.linear(x, weight), 30 * F.linear(x, single), prototype_logits(x, multi, mass), y


def calibrate(batches):
    hits = {name: dict.fromkeys(ALPHAS, 0) for name in ("single", "multiple")}
    for native, single, multi, target in batches:
        for name, proto in (("single", single), ("multiple", multi)):
            for alpha in ALPHAS:
                hits[name][alpha] += int((blend_logits(native, proto, alpha).argmax(1) == target).sum())
    return {name: max(scores, key=scores.get) for name, scores in hits.items()}, hits


def evaluate(batches, alphas, classes, seed):
    names = ("native", "single", "multiple", "native_single_mix", "native_multiple_mix")
    results = {name: [] for name in names}
    labels = []
    for native, single, multi, target in batches:
        scores = (native, single, multi, blend_logits(native, single, alphas["single"]),
                  blend_logits(native, multi, alphas["multiple"]))
        for name, logits in zip(names, scores, strict=True):
            results[name].append(logits.argmax(1))
        labels.append(target)
    targets = torch.cat(labels)
    predictions = {name: torch.cat(parts) for name, parts in results.items()}
    baseline = predictions["native"]
    satellites = device_array(classes[:, 0].astype(np.int64), targets.device)
    counts = torch.bincount(targets, minlength=len(classes))
    base_ok = (baseline == targets).to(torch.int64)
    _, group = np.unique(np.from_dlpack(satellites[targets]), return_inverse=True)
    satellites_counts = np.bincount(group)
    summaries = {}
    for name, pred in predictions.items():
        info = summarize(pred, baseline, targets, satellites, counts)
        delta = np.from_dlpack((pred == targets).to(torch.int64) - base_ok)
        info["top1_delta_satellite_ci95"] = paired_delta_interval(satellites_counts, np.bincount(group, weights=delta), seed)
        summaries[name] = info
    return len(targets), summaries


@torch.no_grad()
def compare(source, cache, checkpoint, seed, smoke=False):
    device = torch.device("cpu")
    classes, _, roles = source_inputs(source, cache)
    model, cfg, updates = load_source_model(source, checkpoint, classes, device)
    if (updates != 846750 or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
            or cfg["input_samples"] != 11000 or cfg["source_roles"] != ["L", "V"] or len(classes) != 3103
            or Path(cfg["cache"]).resolve() != cache):
        raise ValueError("Expected the fixed matched E30 PAMF source checkpoint")
    l_raw, l_labels, _ = roles["L"]
    fit, cal = split_enrollment(l_labels, len(classes), seed, 4 if smoke else 32, 1 if smoke else 8)
    if smoke:
        cal = cal[:64]
    anchors, = encode(model, cfg["arm"], l_raw, fit, device)
    with np.load(source / "L_index.npz", allow_pickle=False) as manifest:
        fit_ticks = manifest["timestamp_ns"][fit]
    single, multi, mass = fit_centers(anchors, l_labels[fit], fit_ticks, len(classes))
    centers = (F.normalize(model.head.weight, dim=1), single, multi, mass)
    calibration, = encode(model, cfg["arm"], l_raw, cal, device)
    cal_targets = device_array(l_labels[cal].astype(np.int64), device)
    alphas, hits = calibrate(score_batches(calibration, cal_targets, *centers))
    print(f"Source L: {len(fit)} fit, {len(cal)} calibrate, alpha {alphas}", flush=True)
    if smoke:
        query, y = calibration[:32], cal_targets[:32]
    else:
        v_raw, v_labels, _ = roles["V"]
        rows = np.flatnonzero(v_labels >= 0)
        query, = encode(model, cfg["arm"], v_raw, rows, device)
        y = device_array(v_labels[rows].astype(np.int64), device)
    n, metrics = evaluate(score_batches(query, y, *centers), alphas, classes, seed)
    return dict(checkpoint=str(checkpoint), updates=updates, smoke=smoke, seed=seed, candidates=len(classes),
                records=n, fit_records=len(fit), calibration_records=len(cal), selected=alphas, calibration_hits=hits,
                prototype_counts=torch.bincount(mass.isfinite().sum(1), minlength=SLOTS + 1).tolist(), methods=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "cache", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare(args.source.resolve(), args.cache.resolve(), args.checkpoint.resolve(), args.seed, args.smoke)
    write_json(args.output, result)
