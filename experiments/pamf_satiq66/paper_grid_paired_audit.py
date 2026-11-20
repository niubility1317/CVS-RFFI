"""Analyze paired all-beam errors from fixed native PAMF source checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_eval import embeddings
from .paper_readout import load_source_model
from .paper_single_packet import argmax_classes
from .paper_train import source_inputs


def paired_delta_interval(counts, delta, seed):
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(64):
        clusters = rng.integers(len(counts), size=(64, len(counts)))
        bootstrap.append(delta[clusters].sum(1) / counts[clusters].sum(1))
    return np.quantile(np.concatenate(bootstrap), [.025, .975]).tolist()


def paired_metrics(labels, classes, prediction_256, prediction_512, seed):
    hit_256 = prediction_256 == labels
    hit_512 = prediction_512 == labels
    counts = np.bincount(labels, minlength=len(classes))
    present = counts > 0
    correct_256 = np.bincount(labels[hit_256], minlength=len(classes))[present]
    correct_512 = np.bincount(labels[hit_512], minlength=len(classes))[present]
    counts = counts[present]
    delta = correct_512 - correct_256
    satellite = classes[:, 0]
    return {
        "packets": len(labels),
        "candidates": len(classes),
        "present_beams": int(present.sum()),
        "correct_256": int(hit_256.sum()),
        "correct_512": int(hit_512.sum()),
        "both_correct": int((hit_256 & hit_512).sum()),
        "only_256_correct": int((hit_256 & ~hit_512).sum()),
        "only_512_correct": int((hit_512 & ~hit_256).sum()),
        "beams_with_hits_256": int(np.count_nonzero(correct_256)),
        "beams_with_hits_512": int(np.count_nonzero(correct_512)),
        "beams_with_more_256_hits": int((delta < 0).sum()),
        "beams_with_more_512_hits": int((delta > 0).sum()),
        "equal_beam_weight_accuracy_256": float((correct_256 / counts).mean()),
        "equal_beam_weight_accuracy_512": float((correct_512 / counts).mean()),
        "top1_delta_beam_bootstrap_ci95": paired_delta_interval(counts, delta, seed),
        "satellite_hit_256": int((satellite[prediction_256] == satellite[labels]).sum()),
        "satellite_hit_512": int((satellite[prediction_512] == satellite[labels]).sum()),
    }


@torch.no_grad()
def audit(source, cache, checkpoints, epoch, *, comparison="grid", smoke=False):
    device = torch.device("cpu")
    classes, _, roles = source_inputs(source, cache)
    raw, labels, _ = roles["L" if smoke else "V"]
    known = np.flatnonzero(labels >= 0)
    if smoke:
        raw, labels = raw[known[:32]], labels[known[:32]]
        known = np.arange(len(labels))
    shaping = comparison == "spectral-shaping"
    grids = (256, 256) if shaping else (256, 512)
    option_differences = {"resume", "native_spectral_gains" if shaping else "native_grid_points"}
    comparable_configs = []
    predicted = []
    expected_updates = epoch * 28225
    for index, (grid, checkpoint) in enumerate(zip(grids, checkpoints, strict=True)):
        model, config, updates = load_source_model(source, checkpoint, classes, device)
        if (config["arm"] != "PAMF_SUP" or config["native_expansion"] != "fused256"
                or config["input_samples"] != 11000 or not config["signed_input"]
                or not config["align_reference_time"] or config.get("native_grid_points", 256) != grid
                or config["source_roles"] != ["L", "V"] or Path(config["cache"]).resolve() != cache
                or config["steps_per_epoch"] != 28225 or updates != expected_updates):
            raise ValueError(f"Expected the matched {grid}-point checkpoint at step {expected_updates}")
        if shaping and (config.get("native_reflect_boundary", False)
                        or bool(config.get("native_spectral_gains")) != bool(index)):
            raise ValueError("Expected periodic baseline followed by the spectral-shaping candidate")
        comparable_configs.append({k: v for k, v in config.items() if k not in option_differences})
        features = torch.from_dlpack(embeddings(model, raw, device)[known])
        weights = F.normalize(model.head.weight, dim=1)
        predicted.append(argmax_classes(features, weights, device))
        print(f"Scored {checkpoint} at {updates} updates", flush=True)
    if comparable_configs[0] != comparable_configs[1]:
        raise ValueError(f"The frozen configurations differ beyond {comparison} and the resume path")
    seed = comparable_configs[0]["evaluation_seed"]
    targets = labels[known]
    metrics = paired_metrics(targets, classes, *predicted, seed)
    result = {"source_epoch": epoch, "updates": expected_updates, "seed": seed,
              "checkpoints": [str(p) for p in checkpoints]}
    if shaping:
        _, satellite_group = np.unique(classes[targets, 0], return_inverse=True)
        total = np.bincount(satellite_group)
        old, new = (np.bincount(satellite_group[guess == targets], minlength=len(total)) for guess in predicted)
        metrics = {k.replace("_256", "_baseline").replace("_512", "_shaping"): v for k, v in metrics.items()}
        result.update(comparison=comparison, smoke=smoke, present_satellites=len(total),
                      satellites_with_more_baseline_hits=int((new < old).sum()),
                      satellites_with_more_shaping_hits=int((new > old).sum()),
                      top1_delta_satellite_bootstrap_ci95=paired_delta_interval(total, new - old, seed))
    return {**result, **metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output", "grid256", "grid512"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--comparison", choices=("grid", "spectral-shaping"), default="grid")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, audit(args.source.resolve(), args.cache.resolve(),
                                 (args.grid256.resolve(), args.grid512.resolve()), args.epoch,
                                 comparison=args.comparison, smoke=args.smoke))
