"""Measure frozen PAMF beam-ranking reliance on bands of the same IQ packet."""

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


def lowpass_masks(physical_hz):
    masks = {"fused": torch.ones_like(physical_hz)}
    for cutoff in (25, 50, 100):
        transition = ((physical_hz - cutoff * 1000) / 5000).clamp(0, 1)
        masks[f"keep_{cutoff}khz"] = .5 * (1 + torch.cos(torch.pi * transition))
    return masks


@torch.no_grad()
def score_cutoffs(model, batches, masks):
    original_cutoff = model.cutoff.clone()
    cutoffs = {name: original_cutoff * mask for name, mask in masks.items()}
    arms = {name: dict(correct=0, gained_vs_fused=0, lost_vs_fused=0,
                      unchanged_top1=0, ce=0.0) for name in masks}
    records = 0
    try:
        for iq, truth in batches:
            records += len(truth)
            model.cutoff.copy_(cutoffs["fused"])
            _, original = model(iq)
            fused_top1 = original.argmax(1)
            fused_ok = fused_top1 == truth
            for name, cutoff in cutoffs.items():
                if name == "fused":
                    scores, pred = original, fused_top1
                else:
                    model.cutoff.copy_(cutoff)
                    _, scores = model(iq)
                    pred = scores.argmax(1)
                ok = pred == truth
                stats = arms[name]
                stats["correct"] += int(ok.sum())
                stats["gained_vs_fused"] += int((ok & ~fused_ok).sum())
                stats["lost_vs_fused"] += int((~ok & fused_ok).sum())
                stats["unchanged_top1"] += int((pred == fused_top1).sum())
                stats["ce"] += float(F.cross_entropy(scores, truth, reduction="sum"))
    finally:
        model.cutoff.copy_(original_cutoff)
    for stats in arms.values():
        stats["ce"] /= records
    return dict(records=records, arms=arms)


@torch.no_grad()
def compare(source, cache, checkpoint, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    device = torch.device("cpu")
    model, config, updates = load_source_model(source, checkpoint, classes, device)
    if (updates != 846750 or config["arm"] != "PAMF_SUP" or config["native_expansion"] != "fused256"
            or model.grid_points != 256 or model.waveform_samples != 11000 or model.reflect_boundary
            or config["source_roles"] != ["L", "V"] or Path(config["cache"]).resolve() != cache):
        raise ValueError("Expected the fixed same-header native PAMF E30 source checkpoint")
    raw, labels, _ = roles["L" if smoke else "V"]
    rows = np.flatnonzero(labels >= 0)
    if smoke:
        rows = rows[:32]
    physical_hz = torch.arange(len(model.cutoff), device=device) * (25_000_000 / model.waveform_samples)
    masks = lowpass_masks(physical_hz)

    def inputs():
        for start in range(0, len(rows), 128):
            ix = rows[start:start + 128]
            iq = model_input(model, device_array(raw[ix, :, :model.waveform_samples], device))
            yield iq, device_array(labels[ix].astype(np.int64), device)

    results = score_cutoffs(model, inputs(), masks)
    return dict(checkpoint=str(checkpoint), updates=updates, candidates=len(classes), smoke=smoke,
                grid_points=model.grid_points, fft_spacing_hz=25_000_000 / model.waveform_samples, **results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "cache", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare(args.source.resolve(), args.cache.resolve(), args.checkpoint.resolve(), args.smoke)
    write_json(args.output, result)
