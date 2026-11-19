"""Diagnose PAMF-SUP's classifier with training-only beam prototypes."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import apply_calibration, encode, fit_calibration
from .paper_eval import binary_scores
from .paper_readout import load_source_model
from .paper_train import source_inputs
from .train import device_array


def enroll_rows(labels, class_count, per_class, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(labels, kind="stable")
    boundaries = np.searchsorted(labels[order], np.arange(class_count + 1))
    rows = []
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        rows.append(rng.choice(order[low:high], min(per_class, high - low), replace=False))
    return np.sort(np.concatenate(rows))


def prototype_weights(features, labels, class_count):
    sums = features.new_zeros(class_count, features.shape[1])
    sums.index_add_(0, labels, features)
    return F.normalize(sums, dim=1)


def readout_vectors(paths, stats, mode):
    if mode == "centered":
        paths = (x - s[0] for x, s in zip(paths, stats, strict=True))
    elif mode == "partial_whiten":
        paths = (apply_calibration(x, s) for x, s in zip(paths, stats, strict=True))
    return F.normalize(torch.cat([F.normalize(x, dim=1) for x in paths], dim=1), dim=1)


def score_head(query, weight, target, satellites, wrong, device):
    classes = len(weight)
    count = torch.bincount(target, minlength=classes)
    correct = torch.zeros_like(count)
    predictions = torch.zeros_like(count)
    within_correct = 0
    genuine, impostor = [], []
    for start in range(0, len(target), 256):
        end = start + 256
        labels = target[start:end]
        scores = F.linear(query[start:end].to(device), weight)
        top = scores.argmax(dim=1)
        predictions += torch.bincount(top, minlength=classes)
        correct += torch.bincount(labels[top == labels], minlength=classes)
        same = satellites[None, :] == satellites[labels, None]
        within_correct += int((scores.masked_fill(~same, -torch.inf).argmax(1) == labels).sum())
        valid = wrong[start:end] >= 0
        genuine.append(scores.gather(1, labels[:, None])[valid, 0].cpu())
        impostor.append(scores.gather(1, wrong[start:end].clamp_min(0)[:, None])[valid, 0].cpu())
    has_queries = count > 0
    return {
        "top1": float(correct.sum() / len(target)),
        "macro_top1": float((correct[has_queries] / count[has_queries]).mean()),
        "within_satellite_top1": within_correct / len(target),
        "predicted_classes": int((predictions > 0).sum()),
        "mode_share": float(predictions.max() / len(target)),
        "same_satellite": binary_scores(np.from_dlpack(torch.cat(genuine)), np.from_dlpack(torch.cat(impostor))),
    }


@torch.no_grad()
def compare(source, cache, checkpoints, seed, per_class, fit_count, device):
    classes, _, roles = source_inputs(source, cache)
    train_iq, train_labels, _ = roles["L"]
    query_iq, query_labels, _ = roles["V"]
    enrolled = enroll_rows(train_labels, len(classes), per_class, seed)
    fitted = np.sort(np.random.default_rng(seed).choice(len(train_iq), fit_count, replace=False))
    queries = np.flatnonzero(query_labels >= 0)
    fit_rows = np.union1d(enrolled, fitted)
    fit_indices = np.searchsorted(fit_rows, fitted)
    enrollment_indices = np.searchsorted(fit_rows, enrolled)
    targets = query_labels[queries]
    rng = np.random.default_rng(seed)
    wrong = np.full(len(queries), -1, dtype=np.int64)
    for identity in np.unique(targets):
        candidates = np.flatnonzero((classes[:, 0] == classes[identity, 0]) & (np.arange(len(classes)) != identity))
        if len(candidates):
            subset = targets == identity
            wrong[subset] = rng.choice(candidates, int(subset.sum()))
    target = device_array(targets.astype(np.int64), device)
    satellites = device_array(classes[:, 0].astype(np.int64), device)
    wrong = device_array(wrong, device)
    enrolled_labels = device_array(train_labels[enrolled].astype(np.int64), torch.device("cpu"))
    results = []
    for checkpoint in checkpoints:
        model, config, updates = load_source_model(source, checkpoint, classes, device)
        if config["arm"] != "PAMF_SUP" or Path(config["cache"]).resolve() != cache.resolve():
            raise ValueError(f"Unexpected PAMF-SUP checkpoint source for {checkpoint}")
        l = encode(model, "PAMF_SUP", train_iq, fit_rows, device)
        v = encode(model, "PAMF_SUP", query_iq, queries, device)
        fused = F.normalize(torch.cat(v, dim=1), dim=1)
        scores = {"native_head": score_head(fused, F.normalize(model.head.weight, dim=1), target,
                                            satellites, wrong, device)}
        stats = [fit_calibration(x[fit_indices]) for x in l]
        enrolled_paths = [x[enrollment_indices] for x in l]
        for tag in ("raw", "centered", "partial_whiten"):
            l_vectors = readout_vectors(enrolled_paths, stats, tag)
            v_vectors = readout_vectors(v, stats, tag)
            width = v[0].shape[1]
            scores[tag] = {}
            for view, columns in (("native_160", slice(0, width)), ("full_rate_160", slice(width, None)),
                                  ("fused_320", slice(None))):
                references = F.normalize(l_vectors[:, columns], dim=1)
                queries_for_view = F.normalize(v_vectors[:, columns], dim=1)
                weight = prototype_weights(references, enrolled_labels, len(classes)).to(device)
                scores[tag][view] = score_head(queries_for_view, weight, target, satellites, wrong, device)
        results.append(dict(checkpoint=str(checkpoint), updates=updates, metrics=scores))
        print(f"Scored {checkpoint.parent.parent.name} at {updates} steps", flush=True)
    return dict(seed=seed, per_class=per_class, labeled_enrollment=len(enrolled), fit_count=fit_count,
                known_validation_records=len(queries), classes=len(classes), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--fit-count", type=int, default=8192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, args.seed,
                                   args.per_class, args.fit_count, torch.device(args.device)))
