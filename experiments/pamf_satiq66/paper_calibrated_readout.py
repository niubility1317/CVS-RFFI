"""Compare frozen SatIQ readouts with feature statistics fitted exclusively on source L."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, write_json
from .paper_eval import metrics, normalize_packet
from .paper_signal_readout import panel_rows, source_panels
from .paper_train import make_model, source_inputs
from .train import device_array


@torch.no_grad()
def encode(model, arm, raw, index, device):
    blocks = []
    samples = getattr(model, "waveform_samples", raw.shape[-1])
    for start in range(0, len(index), 128):
        iq = normalize_packet(device_array(raw[index[start:start + 128], :, :samples], device))
        vectors = (model.embed(iq),) if arm == "SATIQ" else model.encode_paths(iq)
        blocks.append(tuple(value.cpu() for value in vectors))
    return tuple(torch.cat(parts) for parts in zip(*blocks, strict=True))


def fit_calibration(training):
    mean = training.mean(dim=0)
    centered = training - mean
    covariance = centered.T @ centered / len(training)
    eigenvalue, basis = torch.linalg.eigh(covariance)
    eigenvalue = eigenvalue.clamp_min(0)
    scale = (eigenvalue + .2 * eigenvalue.mean() + 1e-12).pow(-.25)
    return mean, basis, scale


def apply_calibration(feature, stats):
    mean, basis, scale = stats
    return ((feature - mean) @ basis) * scale


def fit_transforms(training, validation):
    stats = fit_calibration(training)
    return validation, validation - stats[0], apply_calibration(validation, stats)


@torch.no_grad()
def compare(source, cache, checkpoints, seed, fit_count, device):
    classes, _, roles = source_inputs(source, cache)
    train_iq = roles["L"][0]
    raw, labels, _ = roles["V"]
    panels, cutoff = source_panels(source, labels, classes, seed)
    needed, panels = panel_rows(panels, len(raw))
    fit = np.sort(np.random.default_rng(seed).choice(len(train_iq), fit_count, replace=False))
    results = []
    for checkpoint in checkpoints:
        config = read_json(checkpoint.parent / "resolved_config.json")
        if (Path(config["source"]).resolve() != source.resolve() or Path(config["cache"]).resolve() != cache.resolve()
                or config["source_roles"] != ["L", "V"] or config["test_access"] is not False):
            raise ValueError(f"Unexpected checkpoint source data contract for {checkpoint}")
        arm = config["arm"]
        model = make_model(config, arm, classes).to(device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        updates = state["updates"]
        del state
        train = encode(model, arm, train_iq, fit, device)
        val = encode(model, arm, raw, needed, device)
        transformed = [fit_transforms(a, b) for a, b in zip(train, val, strict=True)]
        outcomes = {}
        for tag, paths in zip(("native", "centered", "partial_whiten"), zip(*transformed, strict=True), strict=True):
            features = torch.cat([F.normalize(x, dim=1) for x in paths], dim=1)
            outcomes[tag] = {name: metrics(np.from_dlpack(features), p, device) for name, p in panels.items()}
        result = {"arm": arm, "checkpoint": str(checkpoint), "updates": updates, "outcomes": outcomes}
        results.append(result)
        print(f"Scored source V for {arm} step {updates} ({checkpoint.parent.parent.name})", flush=True)
    return {"source_train_fit_count": fit_count, "source_validation_records": len(needed), "seed": seed,
            "chronological_cutoff_utc": cutoff, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--fit-count", type=int, default=8192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, args.seed, args.fit_count,
                                   torch.device(args.device)))
