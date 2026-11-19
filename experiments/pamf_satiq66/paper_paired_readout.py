"""Paired satellite-cluster uncertainty for fixed source K16 verification."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import apply_calibration, encode, fit_calibration
from .paper_eval import binary_scores
from .paper_readout import load_source_model
from .paper_signal_readout import panel_rows, source_panels
from .paper_train import source_inputs
from .train import device_array


def same_satellite_scores(features, panel, device):
    references, queries, labels, same_sat, _ = panel
    vectors = F.normalize(features.to(device), dim=1)
    enrolled = vectors[device_array(references, device)].mean(dim=1)
    query = vectors[device_array(queries, device)]
    scores = np.from_dlpack((query @ enrolled.T).cpu())
    valid = np.flatnonzero(same_sat >= 0)
    return scores[valid, labels[valid]], scores[valid, same_sat[valid]]


def bootstrap_auc(genuine, impostor, cluster, draws):
    """Average tied wins and resample whole satellites with the same draw per arm."""
    counts = np.bincount(cluster)
    wins = np.empty((len(counts), len(counts)), dtype=np.float64)
    for other in range(len(counts)):
        scores = np.sort(impostor[cluster == other])
        lower = np.searchsorted(scores, genuine, side="left")
        upper = np.searchsorted(scores, genuine, side="right")
        wins[:, other] = np.bincount(cluster, weights=(lower + upper) * .5, minlength=len(counts))
    return np.einsum("bi,ij,bj->b", draws, wins, draws, optimize=True) / (draws @ counts) ** 2


def interval(values):
    return np.quantile(values, (.025, .975)).tolist()


@torch.no_grad()
def compare(source, cache, checkpoints, seed, fit_count, draws, device):
    classes, _, roles = source_inputs(source, cache)
    train_iq = roles["L"][0]
    raw, labels, _ = roles["V"]
    panels, cutoff = source_panels(source, labels, classes, seed)
    needed, panels = panel_rows(panels, len(raw))
    rng = np.random.default_rng(seed)
    fit = np.sort(rng.choice(len(train_iq), fit_count, replace=False))
    clusters = {}
    for name, (_, queries, _, same_sat, _) in panels.items():
        satellites = classes[labels[needed[queries[same_sat >= 0]]], 0]
        groups, indices = np.unique(satellites, return_inverse=True)
        cluster_draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
        clusters[name] = groups, indices, cluster_draws

    results, bootstraps = [], []
    for checkpoint in checkpoints:
        model, config, step = load_source_model(source, checkpoint, classes, device)
        if Path(config["cache"]).resolve() != cache.resolve():
            raise ValueError("Readout cache must match source checkpoint")
        arm = config["arm"]
        l = encode(model, arm, train_iq, fit, device)
        v = encode(model, arm, raw, needed, device)
        calibrated = [apply_calibration(vectors, fit_calibration(training))
                      for training, vectors in zip(l, v, strict=True)]
        metrics, boot = {}, {}
        for mode, paths in (("raw", v), ("partial_whiten", calibrated)):
            features = torch.cat([F.normalize(x, dim=1) for x in paths], dim=1)
            metrics[mode] = {}
            for name, panel in panels.items():
                genuine, impostor = same_satellite_scores(features, panel, device)
                groups, labels_by_group, cluster_draws = clusters[name]
                sample = bootstrap_auc(genuine, impostor, labels_by_group, cluster_draws)
                metrics[mode][name] = {**binary_scores(genuine, impostor), "satellites": len(groups),
                                       "auc_interval": interval(sample)}
                boot[mode, name] = sample
        result = dict(checkpoint=str(checkpoint), arm=arm, updates=step, metrics=metrics)
        for previous, previous_boot in zip(results, bootstraps, strict=True):
            comparison = dict(against=previous["checkpoint"], delta={})
            for mode, name in boot:
                observed = metrics[mode][name]["auc"] - previous["metrics"][mode][name]["auc"]
                comparison["delta"].setdefault(mode, {})[name] = dict(
                    auc=observed, interval=interval(boot[mode, name] - previous_boot[mode, name]))
            result.setdefault("comparisons", []).append(comparison)
        results.append(result)
        bootstraps.append(boot)
        print(f"Scored {checkpoint.parent.parent.name} step {step}", flush=True)
    return dict(source_train_fit_count=fit_count, source_fit_seed=seed, bootstrap_draws=draws,
                chronological_cutoff_utc=cutoff, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "cache", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--fit-count", type=int, default=8192)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, args.seed,
                                   args.fit_count, args.draws, torch.device(args.device)))
