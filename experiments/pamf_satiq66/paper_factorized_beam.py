"""Test a source-fitted satellite score in the unchanged all-beam decision."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import encode
from .paper_readout import load_source_model
from .paper_satellite_pooling_readout import satellite_evidence
from .paper_sup_prototype_readout import enroll_rows
from .paper_train import source_inputs
from .train import device_array


def fit_satellite_ridge(features, targets, classes):
    x = features.double()
    center = x.mean(dim=0)
    x = x - center
    counts = torch.bincount(targets, minlength=classes)
    if len(counts) != classes or (counts == 0).any():
        raise ValueError("Every satellite needs source fitting records")
    covariance = x.T @ x / len(x)
    cross = x.new_zeros(classes, x.shape[1])
    cross.index_add_(0, targets, x)
    ridge = .1 * covariance.diagonal().mean()
    covariance.diagonal().add_(ridge)
    weight = torch.cholesky_solve(cross.T / len(x), torch.linalg.cholesky(covariance)).T
    return center.float(), weight.float(), (counts.double() / len(x)).float()


def beam_evidence(logits, columns, counts):
    return satellite_evidence(logits, columns, counts) + counts.log()


def adjusted_beam_score(logits, native_satellite, independent_satellite, columns, scale, weight):
    if weight == 0:
        return logits
    return logits + weight * (scale * independent_satellite - native_satellite)[:, columns]


def select_weight(calibration, targets, infer, columns):
    sum_squares = torch.zeros(2, dtype=torch.float64)
    blocks = []
    for batch in calibration.split(256):
        logits, native, independent = infer(batch)
        blocks.append((logits, native, independent))
        for i, s in enumerate((native, independent)):
            sum_squares[i] += (s - s.mean(dim=1, keepdim=True)).square().sum(dtype=torch.float64)
    if not torch.isfinite(sum_squares).all() or (sum_squares <= 0).any():
        raise ValueError("Expected finite, nonzero satellite-score spread on source L")
    scale = float((sum_squares[0] / sum_squares[1]).sqrt())
    outcomes = {}
    for w in (0., .1, .2, .35, .5, 1.):
        predicted = torch.cat([adjusted_beam_score(a, b, c, columns, scale, w).argmax(1) for a, b, c in blocks])
        outcomes[w] = int((predicted == targets).sum())
    return scale, max(outcomes, key=outcomes.get), outcomes


def score(embeddings, targets, infer, columns, scale, selected):
    results = {k: dict(beam_hits=0, winning_satellite_hits=0, corrections=0, regressions=0)
               for k in ("native", "full_factorization", "calibrated_factorization")}
    coarse_hits = dict(native_marginal=0, independent_ridge=0)
    target_satellite = columns[targets]
    for start in range(0, len(targets), 256):
        end = start + 256
        y = targets[start:end]
        sat = target_satellite[start:end]
        logits, marginal, ridge = infer(embeddings[start:end])
        coarse_hits["native_marginal"] += int((marginal.argmax(1) == sat).sum())
        coarse_hits["independent_ridge"] += int((ridge.argmax(1) == sat).sum())
        base_correct = logits.argmax(1) == y
        for (name, out), weight in zip(results.items(), (0., 1., selected), strict=True):
            pred = adjusted_beam_score(logits, marginal, ridge, columns, scale, weight).argmax(1)
            correct = pred == y
            out["beam_hits"] += int(correct.sum())
            out["winning_satellite_hits"] += int((columns[pred] == sat).sum())
            out["corrections"] += int((correct & ~base_correct).sum())
            out["regressions"] += int((~correct & base_correct).sum())
    return dict(beam_heads=results, satellite_heads=coarse_hits)


@torch.no_grad()
def compare(source, cache, checkpoint, seed, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    cpu = torch.device("cpu")
    model, cfg, updates = load_source_model(source, checkpoint, classes, cpu)
    if (updates != 846750 or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
            or cfg["input_samples"] != 11000 or cfg["source_roles"] != ["L", "V"]
            or Path(cfg["cache"]).resolve() != cache):
        raise ValueError("Expected the fixed matched E30 PAMF source checkpoint")
    sats, col, count = np.unique(classes[:, 0], return_inverse=True, return_counts=True)
    columns = device_array(col.astype(np.int64), cpu)
    satellite_beam_counts = device_array(count.astype(np.float32), cpu)
    raw_l, labels_l, _ = roles["L"]
    fitted = enroll_rows(labels_l, len(classes), 2 if smoke else 32, seed)
    feat, = encode(model, cfg["arm"], raw_l, fitted, cpu)
    y = device_array(labels_l[fitted].astype(np.int64), cpu)
    center, satellite_weight, satellite_bias = fit_satellite_ridge(feat, columns[y], len(sats))
    beam_weight = F.normalize(model.head.weight, dim=1)

    def infer(x):
        beam = 30 * F.linear(x, beam_weight)
        return beam, beam_evidence(beam, columns, satellite_beam_counts), F.linear(x - center, satellite_weight, satellite_bias)

    remaining = labels_l.copy()
    remaining[fitted] = -1
    calibration = enroll_rows(remaining, len(classes), 2 if smoke else 8, seed + 1)
    calibrating, = encode(model, cfg["arm"], raw_l, calibration, cpu)
    labels_cal = device_array(labels_l[calibration].astype(np.int64), cpu)
    scale, weight, calibration_hits = select_weight(calibrating, labels_cal, infer, columns)
    print(f"Fitted {len(fitted)}, calibrated {len(calibration)} source L packets, selected weight {weight}", flush=True)
    if smoke:
        embeddings, targets = calibrating[:32], labels_cal[:32]
    else:
        raw_v, labels_v, _ = roles["V"]
        known = np.flatnonzero(labels_v >= 0)
        embeddings, = encode(model, cfg["arm"], raw_v, known, cpu)
        targets = device_array(labels_v[known].astype(np.int64), cpu)
    return dict(checkpoint=str(checkpoint), updates=updates, smoke=smoke, seed=seed, beams=len(classes), satellites=len(sats),
                fitted_records=len(fitted), calibration_records=len(calibration), records=len(targets),
                satellite_scale=scale, selected_weight=weight, calibration_hits=calibration_hits,
                **score(embeddings, targets, infer, columns, scale, weight))


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
