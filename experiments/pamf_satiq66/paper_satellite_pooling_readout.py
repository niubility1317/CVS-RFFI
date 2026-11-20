"""Evaluate source class ranks after pooling model evidence by registered satellite."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_eval import model_input
from .paper_readout import load_source_model
from .paper_train import source_inputs
from .train import device_array


def satellite_evidence(logits, satellite_columns, class_counts):
    stable = logits - logits.amax(dim=1, keepdim=True)
    grouped = logits.new_zeros(len(logits), len(class_counts))
    grouped.scatter_add_(1, satellite_columns.expand(len(logits), -1), stable.exp())
    return (grouped / class_counts).log()


@torch.no_grad()
def score_head(model, raw, labels, satellites, device):
    rows = np.flatnonzero(labels >= 0)
    _, columns, counts = np.unique(satellites, return_inverse=True, return_counts=True)
    columns = device_array(columns.astype(np.int64), device)
    counts = device_array(counts.astype(np.float32), device)
    weight = F.normalize(model.head.weight, dim=1)
    correct = torch.zeros(2, device=device, dtype=torch.int64)
    satellite_correct = torch.zeros(3, device=device, dtype=torch.int64)
    loss = torch.zeros(2, device=device, dtype=torch.float64)
    marginal_correct = torch.zeros((), device=device, dtype=torch.int64)
    marginal_loss = torch.zeros((), device=device, dtype=torch.float64)
    conditional_correct = torch.zeros((), device=device, dtype=torch.int64)
    log_counts = counts.log()
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        iq = model_input(model, device_array(raw[batch, :, :model.waveform_samples], device))
        logits = 30 * F.linear(model.embed(iq), weight)
        pool = satellite_evidence(logits, columns, counts)
        adjusted = logits + pool[:, columns]
        targets = device_array(labels[batch].astype(np.int64), device)
        target_satellites = columns[targets]
        satellite_correct[2] += (pool.argmax(dim=1) == target_satellites).sum()
        # Divide-by-count pooling serves the older reranking audit. Undo it for
        # the probability mass optimized by the satellite-marginal loss.
        marginal = pool + log_counts
        marginal_correct += (marginal.argmax(dim=1) == target_satellites).sum()
        marginal_loss += F.cross_entropy(marginal, target_satellites, reduction="sum")
        same_satellite = columns[None, :] == target_satellites[:, None]
        conditional = logits.masked_fill(~same_satellite, -torch.inf)
        conditional_correct += (conditional.argmax(dim=1) == targets).sum()
        for index, scores in enumerate((logits, adjusted)):
            prediction = scores.argmax(dim=1)
            correct[index] += (prediction == targets).sum()
            satellite_correct[index] += (columns[prediction] == target_satellites).sum()
            loss[index] += F.cross_entropy(scores, targets, reduction="sum")
    return dict(
        records=len(rows), pooled_satellite_top1=int(satellite_correct[2]) / len(rows),
        marginal_satellite_top1=int(marginal_correct) / len(rows),
        marginal_satellite_ce=float(marginal_loss) / len(rows),
        conditional_beam_top1=int(conditional_correct) / len(rows),
        conditional_beam_ce=float(loss[0] - marginal_loss) / len(rows),
        readouts={
            name: dict(correct=int(correct[i]), beam_top1=int(correct[i]) / len(rows),
                       first_satellite_top1=int(satellite_correct[i]) / len(rows), ce=float(loss[i]) / len(rows))
            for i, name in enumerate(("raw", "pooled"))},
    )


def compare(source, cache, checkpoints, device):
    classes, _, roles = source_inputs(source, cache)
    raw, labels, _ = roles["V"]
    results = []
    for checkpoint in checkpoints:
        model, config, updates = load_source_model(source, checkpoint, classes, device)
        if (config["arm"] != "PAMF_SUP" or "native_expansion" not in config
                or Path(config["cache"]).resolve() != cache.resolve()):
            raise ValueError("Expected native PAMF-SUP with its registered source cache")
        scores = score_head(model, raw, labels, classes[:, 0], device)
        results.append(dict(checkpoint=str(checkpoint), updates=updates, **scores))
        print(checkpoint.parent.parent.name, updates, scores, flush=True)
    return dict(role="V", fit_data=None, classes=len(classes), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, torch.device(args.device)))
