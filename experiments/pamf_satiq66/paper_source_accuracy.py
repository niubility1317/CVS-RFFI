"""Measure the frozen PAMF beam classifier on the full source L and V."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_eval import normalize_packet
from .paper_readout import load_source_model
from .paper_train import source_inputs
from .train import device_array


@torch.no_grad()
def classifier_accuracy(model, raw, labels, device, block=256):
    known = int(np.count_nonzero(labels >= 0))
    weight = F.normalize(model.head.weight, dim=1)
    samples = getattr(model, "waveform_samples", raw.shape[-1])
    hits = torch.zeros((), dtype=torch.int64, device=device)
    loss = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, len(labels), block):
        end = start + block
        iq = normalize_packet(device_array(raw[start:end, :, :samples], device))
        target = device_array(labels[start:end].astype(np.int64), device)
        scores = 30 * F.linear(model.embed(iq), weight)
        hits += (scores.argmax(dim=1) == target).sum()
        loss += F.cross_entropy(scores, target, reduction="sum", ignore_index=-1)
    correct = int(hits)
    return dict(records=len(labels), known_records=known, correct=correct, top1=correct / known, ce=float(loss) / known)


def compare(source, cache, checkpoints, device):
    classes, _, roles = source_inputs(source, cache)
    results = []
    for checkpoint in checkpoints:
        model, config, steps = load_source_model(source, checkpoint, classes, device)
        if config["arm"] == "SATIQ" or Path(config["cache"]).resolve() != cache.resolve():
            raise ValueError("Require a PAMF classifier and its original source cache")
        row = dict(checkpoint=str(checkpoint), updates=steps, accuracy={})
        for role in ("L", "V"):
            score = classifier_accuracy(model, *roles[role][:2], device)
            row["accuracy"][role] = score
            print(f"{checkpoint.parent.parent.name} {steps} {role} {score}", flush=True)
        results.append(row)
    return dict(classes=len(classes), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, torch.device(args.device)))
