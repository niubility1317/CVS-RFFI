"""Compare frozen PAMF heads using source L class means and pooled residuals."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import encode
from .paper_readout import load_source_model
from .paper_sup_prototype_readout import enroll_rows
from .paper_train import source_inputs
from .train import device_array


def fit_head(features, labels, classes):
    training = features.double()
    counts = torch.bincount(labels, minlength=classes)
    if len(counts) != classes or (counts < 2).any():
        raise ValueError("Covariance head needs at least two source anchors for every candidate")
    center = training.mean(dim=0)
    centered = training - center
    means = training.new_zeros(classes, training.shape[1])
    means.index_add_(0, labels, centered)
    means /= counts[:, None]
    residuals = centered - means[labels]
    covariance = residuals.T @ residuals / (len(residuals) - classes)
    average_variance = covariance.diagonal().mean()
    if not torch.isfinite(covariance).all() or average_variance <= 0:
        raise ValueError("No finite within-beam covariance from source embeddings")
    covariance = .9 * covariance + .1 * average_variance * torch.eye(len(covariance), dtype=covariance.dtype)
    precision_means = torch.cholesky_solve(means.T, torch.linalg.cholesky(covariance)).T
    bias = -.5 * (means * precision_means).sum(dim=1)
    return center.float(), F.normalize(means + center, dim=1).float(), precision_means.float(), bias.float()


def predictions(query, weight, bias=None):
    return torch.cat([F.linear(block, weight, bias).argmax(dim=1) for block in query.split(256)])


def summarize(predicted, baseline, targets, satellites, class_counts):
    match = predicted == targets
    base_match = baseline == targets
    class_hits = torch.bincount(targets[match], minlength=len(satellites))
    present = class_counts > 0
    return dict(hits=int(match.sum()), top1=float(match.double().mean()),
                macro_top1=float((class_hits[present] / class_counts[present]).mean()),
                satellite_hits=int((satellites[predicted] == satellites[targets]).sum()),
                wins_vs_native=int((match & ~base_match).sum()), losses_vs_native=int((base_match & ~match).sum()))


@torch.no_grad()
def compare(source, cache, checkpoint, seed, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    device = torch.device("cpu")
    model, cfg, updates = load_source_model(source, checkpoint, classes, device)
    if (updates != 564500 or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
            or cfg["input_samples"] != 11000 or cfg["source_roles"] != ["L", "V"]
            or Path(cfg["cache"]).resolve() != cache):
        raise ValueError("Expected the matched E20 native PAMF source checkpoint")
    l_raw, l_labels, _ = roles["L"]
    fit_rows = enroll_rows(l_labels, len(classes), 2 if smoke else 64, seed)
    train, = encode(model, cfg["arm"], l_raw, fit_rows, device)
    fit_target = device_array(l_labels[fit_rows].astype(np.int64), device)
    print(f"Encoded {len(fit_rows)} labeled source anchors", flush=True)
    center, cos, weight, bias = fit_head(train, fit_target, len(classes))
    if smoke:
        query, targets = train[:64], fit_target[:64]
    else:
        v_raw, v_labels, _ = roles["V"]
        known = np.flatnonzero(v_labels >= 0)
        query, = encode(model, cfg["arm"], v_raw, known, device)
        targets = device_array(v_labels[known].astype(np.int64), device)
    satellites = device_array(classes[:, 0].astype(np.int64), device)
    native = predictions(query, F.normalize(model.head.weight, dim=1))
    counts = torch.bincount(targets, minlength=len(classes))
    alternatives = dict(native=native, raw_anchor_cosine=predictions(query, cos),
                        residual_covariance=predictions(query - center, weight, bias))
    return dict(checkpoint=str(checkpoint), updates=updates, smoke=smoke, seed=seed, records=len(targets),
                candidates=len(classes), anchors=len(fit_rows), covariance_shrinkage=.1,
                methods={name: summarize(p, native, targets, satellites, counts) for name, p in alternatives.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "cache", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare(args.source.resolve(), args.cache.resolve(), args.checkpoint.resolve(), args.seed, args.smoke)
    write_json(args.output, result)
