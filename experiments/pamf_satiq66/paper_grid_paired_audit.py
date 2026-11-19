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


def paired_metrics(labels, classes, prediction_256, prediction_512, seed):
    hit_256 = prediction_256 == labels
    hit_512 = prediction_512 == labels
    counts = np.bincount(labels, minlength=len(classes))
    present = counts > 0
    correct_256 = np.bincount(labels[hit_256], minlength=len(classes))[present]
    correct_512 = np.bincount(labels[hit_512], minlength=len(classes))[present]
    counts = counts[present]
    delta = correct_512 - correct_256
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(64):
        beam_indices = rng.integers(len(counts), size=(64, len(counts)))
        bootstrap.append(delta[beam_indices].sum(1) / counts[beam_indices].sum(1))
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
        "top1_delta_beam_bootstrap_ci95": np.quantile(np.concatenate(bootstrap), [.025, .975]).tolist(),
        "satellite_hit_256": int((satellite[prediction_256] == satellite[labels]).sum()),
        "satellite_hit_512": int((satellite[prediction_512] == satellite[labels]).sum()),
    }


@torch.no_grad()
def audit(source, cache, checkpoints, epoch):
    device = torch.device("cpu")
    classes, _, roles = source_inputs(source, cache)
    raw, labels, _ = roles["V"]
    known = labels >= 0
    comparable_configs = []
    predicted = []
    expected_updates = epoch * 28225
    for grid, checkpoint in zip((256, 512), checkpoints, strict=True):
        model, config, updates = load_source_model(source, checkpoint, classes, device)
        if (config["arm"] != "PAMF_SUP" or config["native_expansion"] != "fused256"
                or config["input_samples"] != 11000 or not config["signed_input"]
                or not config["align_reference_time"] or config.get("native_grid_points", 256) != grid
                or config["source_roles"] != ["L", "V"] or Path(config["cache"]).resolve() != cache
                or config["steps_per_epoch"] != 28225 or updates != expected_updates):
            raise ValueError(f"Expected the matched {grid}-point checkpoint at step {expected_updates}")
        comparable_configs.append({k: v for k, v in config.items() if k not in ("native_grid_points", "resume")})
        features = torch.from_dlpack(embeddings(model, raw, device)[known])
        weights = F.normalize(model.head.weight, dim=1)
        predicted.append(argmax_classes(features, weights, device))
        print(f"Scored {grid} points at {updates} updates", flush=True)
    if comparable_configs[0] != comparable_configs[1]:
        raise ValueError("The two frozen source configurations differ beyond the grid and resume path")
    seed = comparable_configs[0]["evaluation_seed"]
    return {"source_epoch": epoch, "updates": expected_updates, "seed": seed,
            "checkpoints": [str(p) for p in checkpoints],
            **paired_metrics(labels[known], classes, *predicted, seed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output", "grid256", "grid512"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, audit(args.source.resolve(), args.cache.resolve(),
                                 (args.grid256.resolve(), args.grid512.resolve()), args.epoch))
