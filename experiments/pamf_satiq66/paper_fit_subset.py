"""Check if the native PAMF classifier can fit a fixed, small labeled source subset."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, write_json
from .paper_eval import normalize_packet
from .paper_native_pamf import native_supervised_loss
from .paper_pamf import branch_prototype_loss
from .paper_source_accuracy import classifier_accuracy
from .paper_train import SourceBatches, make_model, source_inputs
from .train import device_array


MILESTONES = (0, 100, 400, 800, 1600)


def fixed_source_rows(labels, seed):
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(np.bincount(labels[labels >= 0]) >= 16)
    beams = rng.choice(eligible, 64, replace=False)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == beam), 16, replace=False) for beam in beams]))


def train_subset(config, classes, iq, labels, original, class_count, device, full_objective):
    model = make_model(config, "PAMF_SUP", classes).to(device)
    satellites = device_array(classes[:, 0].astype(np.int64), device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["pamf_lr"], eps=1e-7)
    sampler = SourceBatches(original, np.bincount(original, minlength=class_count), config["sampling_seed"])
    measurements = []

    def record(step):
        model.eval()
        result = classifier_accuracy(model, iq, labels, device)
        measurements.append(dict(updates=step, **result))
        print("full_objective", full_objective, measurements[-1], flush=True)
        model.train()

    record(0)
    step = 0
    while step < MILESTONES[-1]:
        for indices in sampler.epoch():
            x = normalize_packet(device_array(iq[indices], device))
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
            if step in MILESTONES:
                record(step)
            if step == MILESTONES[-1]:
                break
    return measurements


def run(source, cache, config, device):
    classes, counts, roles = source_inputs(source, cache)
    raw, labels, original = roles["L"]
    rows = fixed_source_rows(labels, config["sampling_seed"])
    iq = raw[rows, :, :8000]
    config = {**config, "native_expansion": "native160"}
    fit_labels = labels[rows]
    results = {}
    for name, full in (("pamf_sup", True), ("global_ce_diagnostic", False)):
        results[name] = train_subset(config, classes, iq, fit_labels, original[rows], len(counts), device, full)
    return dict(source_role="L", classifier_classes=len(classes), fixed_beams=64, packets_per_beam=16,
                fixed_satellites=int(np.unique(classes[fit_labels, 0]).size), model_seed=config["model_seed"],
                sampling_seed=config["sampling_seed"], source_rows=rows.tolist(), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "config", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, run(args.source, args.cache, read_json(args.config), torch.device(args.device)))
