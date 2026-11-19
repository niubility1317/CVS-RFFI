"""Check source-mean score subtraction for the frozen native SatIQ class head."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import encode
from .paper_readout import load_source_model
from .paper_source_accuracy import classifier_accuracy
from .paper_train import source_inputs


@torch.no_grad()
def compare(source, cache, checkpoints, device, seed=2026111907):
    classes, _, roles = source_inputs(source, cache)
    train = roles["L"][0]
    val, labels, _ = roles["V"]
    rows = np.sort(np.random.default_rng(seed).choice(len(train), 8192, replace=False))
    results = []
    for checkpoint in checkpoints:
        model, config, updates = load_source_model(source, checkpoint, classes, device)
        if (config["arm"] != "PAMF_SUP" or "native_expansion" not in config
                or Path(config["cache"]).resolve() != cache.resolve()):
            raise ValueError("Expected native PAMF-SUP and its original source cache")
        (features,) = encode(model, config["arm"], train, rows, device)
        feature_mean = features.mean(dim=0).to(device)
        means = 30 * F.linear(feature_mean, F.normalize(model.head.weight, dim=1))
        raw = classifier_accuracy(model, val, labels, device, head_diagnostics=True)
        centered = classifier_accuracy(model, val, labels, device, head_diagnostics=True, logit_offset=means)
        row = dict(checkpoint=str(checkpoint), updates=updates, raw=raw, source_centered=centered,
                   fit_logit_offset=means.tolist())
        results.append(row)
        print(checkpoint.parent.parent.name, updates, "raw", raw, "source_centered", centered, flush=True)
    return dict(fit_source_role="L", fit_count=len(rows), fit_seed=seed, evaluation_role="V", results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, torch.device(args.device)))
