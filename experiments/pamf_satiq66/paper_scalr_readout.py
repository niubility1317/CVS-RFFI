"""Compare prototypes with support-calibrated local ridge on frozen SatIQ source panels."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase2_practical.milkr.support_calibrated import coefficients, decompose, make_kernel

from .data import write_json
from .paper_calibrated_readout import apply_calibration, encode, fit_calibration
from .paper_eval import binary_scores
from .paper_readout import load_source_model
from .paper_signal_readout import panel_rows, source_panels
from .paper_train import source_inputs
from .train import device_array


RIDGES = (.01, .1, 1.)
LOCAL_CLASSES = 4


def fit_local(support):
    """One support-only comparison per beam against its three nearest prototypes."""
    classes, shots, width = support.shape
    if classes < LOCAL_CLASSES or shots < 2:
        raise ValueError("Local ridge requires at least four classes and multi-shot support")
    vectors = support.reshape(-1, width).astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
    prototypes = vectors.reshape(classes, shots, width).mean(axis=1)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(1e-12)
    similarity = prototypes @ prototypes.T
    np.fill_diagonal(similarity, -np.inf)
    others = np.argsort(-similarity, axis=1, kind="stable")[:, :LOCAL_CLASSES - 1]
    groups = np.column_stack([np.arange(classes), others])
    indices = (groups[:, :, None] * shots + np.arange(shots)).reshape(classes, -1)
    local_labels = np.repeat(np.arange(LOCAL_CLASSES), shots)
    different = local_labels[:, None] != local_labels[None, :]
    alphas, offsets, taus, multiscale, choices = [], [], [], [], []
    for group in indices:
        local = vectors[group]
        sim = local @ local.T
        d = np.maximum(2 - 2 * sim, 0)
        tau = max(float(np.median(np.min(np.where(different, d, np.inf), axis=1))), 1e-6)
        best = None
        for scales in ("single", "multi"):
            parts = decompose(make_kernel(sim, tau, 0, scales), local_labels)
            for ridge in RIDGES:
                alpha, intercept, loo = coefficients(parts, ridge)
                loo_top1 = float((loo.argmax(1) == local_labels).mean())
                loss = float(((loo - parts[2]) ** 2).mean())
                rank = (loo_top1, -loss, ridge)
                if best is None or rank > best[0]:
                    best = rank, alpha[:, 0], intercept[0], scales
        rank, alpha, offset, scales = best
        alphas.append(alpha)
        offsets.append(offset)
        taus.append(tau)
        multiscale.append(scales == "multi")
        choices.append(dict(loo_top1=rank[0], ridge=rank[2], scales=scales))
    return indices, np.asarray(alphas), np.asarray(offsets), np.asarray(taus), np.asarray(multiscale), choices


def score_all(support, queries, fitted):
    indices, alpha, offset, tau, multi, _ = fitted
    device = queries.device
    indices = device_array(indices, device)
    alpha, offset, tau = [device_array(v.astype(np.float32), device) for v in (alpha, offset, tau)]
    multi = torch.tensor(multi.tolist(), dtype=torch.bool, device=device)
    bank = support.flatten(0, 1)
    proto, ridge = [], []
    for block in queries.split(64):
        proto.append(torch.mm(block, support.mean(1).T).cpu())
        distance = (2 - 2 * (block @ bank.T)[:, indices]).clamp_min_(0)
        scaled = -distance / tau[None, :, None]
        base = scaled.exp()
        kernel = torch.where(multi[None, :, None], (base + (4 * scaled).exp() + (.25 * scaled).exp()) / 3, base)
        ridge.append(((kernel * alpha).sum(2) + offset).cpu())
    return [np.from_dlpack(torch.cat(parts)) for parts in (proto, ridge)]


def panel_metrics(scores, labels, same_satellite, other_satellite):
    rows = np.arange(len(labels))
    genuine = scores[rows, labels]
    result = dict(top1=float((scores.argmax(1) == labels).mean()),
                  correct=int((scores.argmax(1) == labels).sum()))
    for tag, negative in (("same_satellite", same_satellite), ("different_satellite", other_satellite)):
        valid = negative >= 0
        result[tag] = binary_scores(genuine[valid], scores[rows[valid], negative[valid]])
    return result


def evaluate(features, panel, device):
    ref, query, labels, same_satellite, other_satellite = panel
    if np.unique(ref).size != ref.size or np.intersect1d(ref, query).size:
        raise ValueError("Support requires distinct physical packets disjoint from the queries")
    features = F.normalize(features.to(device), dim=1)
    support = features[device_array(ref, device)]
    start = time.monotonic()
    fitted = fit_local(np.from_dlpack(support.cpu()))
    seconds = time.monotonic() - start
    choices = fitted[-1]
    scores = score_all(support, features[device_array(query, device)], fitted)
    results = {name: panel_metrics(score, labels, same_satellite, other_satellite)
               for name, score in zip(("prototype", "support_local_ridge"), scores, strict=True)}
    return dict(reference_classes=len(ref), packets_per_class=ref.shape[1], query_records=len(query),
                fit_seconds=seconds, mean_local_support_loo=float(np.mean([r["loo_top1"] for r in choices])),
                selected_ridges={str(lam): sum(r["ridge"] == lam for r in choices) for lam in RIDGES},
                multiscale_classes=sum(r["scales"] == "multi" for r in choices), **results)


@torch.no_grad()
def compare(source, cache, checkpoints, seed, device):
    classes, _, roles = source_inputs(source, cache)
    raw, labels, _ = roles["V"]
    panels, cutoff = source_panels(source, labels, classes, seed)
    needed, panels = panel_rows(panels, len(raw))
    training_raw = roles["L"][0]
    fit = np.sort(np.random.default_rng(seed).choice(len(training_raw), 8192, replace=False))
    results = []
    for checkpoint in checkpoints:
        model, config, step = load_source_model(source, checkpoint, classes, device)
        if Path(config["cache"]).resolve() != cache.resolve():
            raise ValueError("Readout cache must match source checkpoint")
        arm = config["arm"]
        l = encode(model, arm, training_raw, fit, device)
        v = encode(model, arm, raw, needed, device)
        white = [apply_calibration(x, fit_calibration(y)) for x, y in zip(v, l, strict=True)]
        output = {}
        for transform, paths in (("raw", v), ("partial_whiten", white)):
            features = torch.cat([F.normalize(path, dim=1) for path in paths], dim=1)
            output[transform] = {}
            for name, panel in panels.items():
                row = evaluate(features, panel, device)
                output[transform][name] = row
                print(step, checkpoint.parent.parent.name, transform, name,
                      row["prototype"]["top1"], row["support_local_ridge"]["top1"], flush=True)
        results.append(dict(checkpoint=str(checkpoint), updates=step, results=output))
    return dict(source_fit_count=len(fit), source_fit_seed=seed, chronology_cutoff=cutoff, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, args.seed, torch.device(args.device)))
