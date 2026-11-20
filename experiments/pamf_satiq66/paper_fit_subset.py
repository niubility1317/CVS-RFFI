"""Check if the native PAMF classifier can fit a fixed, small labeled source subset."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, write_json
from .paper_eval import model_input
from .paper_native_pamf import native_supervised_loss
from .paper_pamf import branch_prototype_loss
from .paper_source_accuracy import classifier_accuracy
from .paper_train import SourceBatches, make_model, source_inputs
from .train import device_array


MILESTONES = (0, 100, 400, 800, 1600)
LR_MILESTONES = (*MILESTONES, 3200)
LR_CUTOFF_NS = np.datetime64("2023-04-02T00:00:00", "ns").astype(np.int64)


def fixed_source_rows(labels, seed):
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(np.bincount(labels[labels >= 0]) >= 16)
    beams = rng.choice(eligible, 64, replace=False)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == beam), 16, replace=False) for beam in beams]))


def temporal_source_rows(labels, timestamps, seed):
    """Enrollment and later packets share beams and have disjoint time periods."""
    known = labels >= 0
    earlier = known & (timestamps < LR_CUTOFF_NS)
    later = known & ~earlier
    counts = [np.bincount(labels[mask], minlength=int(labels.max()) + 1) for mask in (earlier, later)]
    eligible = np.flatnonzero((counts[0] >= 16) & (counts[1] >= 16))
    rng = np.random.default_rng(seed)
    beams = rng.choice(eligible, 64, replace=False)
    return tuple(np.sort(np.concatenate([rng.choice(np.flatnonzero(mask & (labels == beam)), 16, replace=False)
                                         for beam in beams])) for mask in (earlier, later))


def train_subset(config, classes, iq, labels, original, class_count, device, full_objective,
                 holdout=None, milestones=MILESTONES):
    model = make_model(config, "PAMF_SUP", classes).to(device)
    satellites = device_array(classes[:, 0].astype(np.int64), device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["pamf_lr"], eps=1e-7)
    sampler = SourceBatches(original, np.bincount(original, minlength=class_count), config["sampling_seed"])
    measurements = []

    def record(step):
        model.eval()
        result = classifier_accuracy(model, iq, labels, device)
        if holdout is not None:
            result["later_source"] = classifier_accuracy(model, *holdout, device)
        measurements.append(dict(updates=step, **result))
        print("full_objective", full_objective, measurements[-1], flush=True)
        model.train()

    record(0)
    step = 0
    while step < milestones[-1]:
        for indices in sampler.epoch():
            x = model_input(model, device_array(iq[indices], device))
            y = device_array(labels[indices].astype(np.int64), device)
            optimizer.zero_grad(set_to_none=True)
            if full_objective:
                _, logits, branch_logits, branches = model(x, view_logits=True)
                loss, _ = native_supervised_loss(logits, branch_logits, y, satellites, config["pamf_sup"])
                loss = loss + config["branch_prototype_weight"] * branch_prototype_loss(branches)
            else:
                _, logits = model(x)
                loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite subset objective at update {step + 1}")
            loss.backward()
            optimizer.step()
            step += 1
            if step in milestones:
                record(step)
            if step == milestones[-1]:
                break
    return measurements


def run(source, cache, config, device, full_only=False):
    classes, counts, roles = source_inputs(source, cache)
    raw, labels, original = roles["L"]
    rows = fixed_source_rows(labels, config["sampling_seed"])
    iq = raw[rows, :, :8000]
    config = {**config, "native_expansion": "native160"}
    fit_labels = labels[rows]
    args = config, classes, iq, fit_labels, original[rows], len(counts), device
    results = {"pamf_sup": train_subset(*args, True)}
    if not full_only:
        results["global_ce_diagnostic"] = train_subset(*args, False)
    return dict(source_role="L", classifier_classes=len(classes), fixed_beams=64, packets_per_beam=16,
                fixed_satellites=int(np.unique(classes[fit_labels, 0]).size), model_seed=config["model_seed"],
                sampling_seed=config["sampling_seed"], source_rows=rows.tolist(), results=results)


def learning_rate_study(source, cache, config, device):
    if (config["native_waveform_samples"] != 11000 or not config["align_reference_time"]
            or config["augmentation"] != "none" or config["pamf_lr"] != .0003):
        raise ValueError("Expected the matched aligned-header PAMF baseline")
    config = {**config, "native_expansion": "fused256"}
    classes, counts, roles = source_inputs(source, cache)
    raw, labels, original = roles["L"]
    with np.load(source / "L_index.npz", allow_pickle=False) as manifest:
        timestamps = manifest["timestamp_ns"]
    fit, future = temporal_source_rows(labels, timestamps, config["sampling_seed"])
    args = classes, raw[fit], labels[fit], original[fit], len(counts), device, True
    holdout = (raw[future], labels[future])
    results = []
    for lr in (config["pamf_lr"], .001):
        print(f"Learning rate {lr:g}; all outcomes use the matched source subsets", flush=True)
        metrics = train_subset({**config, "pamf_lr": lr}, *args, holdout=holdout, milestones=LR_MILESTONES)
        results.append(dict(learning_rate=lr, measurements=metrics))
    return dict(source_role="L", classifier_classes=len(classes), fixed_beams=64, packets_per_beam=16,
                cutoff_utc="2023-04-02T00:00:00Z", model_seed=config["model_seed"],
                sampling_seed=config["sampling_seed"], training_rows=fit.tolist(), later_rows=future.tolist(),
                results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "config", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--full-only", action="store_true")
    modes.add_argument("--learning-rate-study", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    common = args.source, args.cache, read_json(args.config), torch.device(args.device)
    result = learning_rate_study(*common) if args.learning_rate_study else run(*common, args.full_only)
    write_json(args.output, result)
