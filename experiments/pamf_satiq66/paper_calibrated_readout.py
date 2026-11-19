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
from .wide_model import FEATURES


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


def evaluate_panels(features, panels, device):
    rows = np.from_dlpack(features)
    return {name: metrics(rows, panel, device) for name, panel in panels.items()}


def feature_geometry(vectors):
    unit = F.normalize(vectors, dim=1)
    n = len(unit)
    off_diagonal_cosine = (unit.sum(0).square().sum() - unit.square().sum()) / (n * (n - 1))
    centered = unit - unit.mean(0)
    covariance = centered.T @ centered / n
    trace_squared = covariance.diag().sum().square()
    return {"dimensions": unit.shape[1], "mean_pair_cosine": float(off_diagonal_cosine),
            "effective_dimensions": float(trace_squared / covariance.square().sum().clamp_min(1e-20))}


def native_components(training, validation, panels, device):
    columns = {"original_160": slice(0, FEATURES)}
    if validation.shape[1] > FEATURES:
        columns["expansion"] = slice(FEATURES, None)
    calibrated = {}
    result = {}
    for name, index in columns.items():
        fit = training[:, index]
        vectors = validation[:, index]
        whitened = apply_calibration(vectors, fit_calibration(fit))
        calibrated[name] = F.normalize(whitened, dim=1)
        result[name] = {
            "geometry": feature_geometry(vectors),
            "raw": evaluate_panels(vectors, panels, device),
            "partial_whiten": evaluate_panels(calibrated[name], panels, device),
        }
    if "expansion" in calibrated:
        # Keep the expansion at its training-time squared norm ratio, 0.25:1.
        combined = torch.cat((calibrated["original_160"], .5 * calibrated["expansion"]), dim=1)
        result["block_calibrated_fusion"] = evaluate_panels(combined, panels, device)
    return result


@torch.no_grad()
def compare(source, cache, checkpoints, seed, fit_count, device, analyze_native_components=False):
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
            outcomes[tag] = evaluate_panels(features, panels, device)
        result = {"arm": arm, "checkpoint": str(checkpoint), "updates": updates, "outcomes": outcomes}
        if analyze_native_components:
            if "native_expansion" not in config:
                raise ValueError(f"Native component scoring requires an original-feature-only checkpoint: {checkpoint}")
            result["native_expansion"] = config["native_expansion"]
            result["components"] = native_components(train[0], val[0], panels, device)
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
    parser.add_argument("--native-components", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, args.seed, args.fit_count,
                                   torch.device(args.device), args.native_components))
