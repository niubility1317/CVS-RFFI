"""Test whether the frozen model's high-band dependence changes with time distance."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_band_evidence import lowpass_masks
from .paper_eval import model_input
from .paper_readout import load_source_model
from .paper_time_gap_audit import GAP_EDGES_SECONDS, nearest_seconds
from .paper_train import source_inputs
from .train import device_array


def within_beam_delta(labels, gap, values, seed):
    near, far = gap <= 5, gap > 30
    size = int(labels.max()) + 1
    nn = np.bincount(labels[near], minlength=size)
    nf = np.bincount(labels[far], minlength=size)
    paired = (nn >= 2) & (nf >= 2)
    if not paired.any():
        return None
    near_mean = np.bincount(labels[near], weights=values[near], minlength=size)[paired] / nn[paired]
    far_mean = np.bincount(labels[far], weights=values[far], minlength=size)[paired] / nf[paired]
    weights = np.minimum(nn[paired], nf[paired])
    delta = far_mean - near_mean
    samples = np.random.default_rng(seed).integers(len(delta), size=(2048, len(delta)))
    bootstrap = (delta * weights)[samples].sum(1) / weights[samples].sum(1)
    return dict(beams=int(paired.sum()), records_per_end=int(weights.sum()),
                near=float(np.average(near_mean, weights=weights)), far=float(np.average(far_mean, weights=weights)),
                far_minus_near=float(np.average(delta, weights=weights)),
                beam_bootstrap_ci95=np.quantile(bootstrap, [.025, .975]).tolist())


def describe_gap(mask, original_ok, masked_ok, original_ce, masked_ce):
    original, masked = original_ok[mask], masked_ok[mask]
    orig_loss, mask_loss = original_ce[mask], masked_ce[mask]
    return dict(records=int(mask.sum()), original_correct=int(original.sum()), masked_correct=int(masked.sum()),
                only_original=int((original & ~masked).sum()), only_masked=int((masked & ~original).sum()),
                unchanged_correctness=int((original == masked).sum()),
                original_ce=float(orig_loss.mean(dtype=np.float64)) if len(orig_loss) else None,
                masked_ce=float(mask_loss.mean(dtype=np.float64)) if len(mask_loss) else None,
                masked_minus_original_ce=float((mask_loss - orig_loss).mean(dtype=np.float64)) if len(mask_loss) else None)


@torch.no_grad()
def measure_predictions(model, raw, labels, rows, device):
    original_cutoff = model.cutoff.clone()
    physical_hz = torch.arange(len(original_cutoff), device=device) * (25_000_000 / model.waveform_samples)
    filtered = original_cutoff * lowpass_masks(physical_hz)["keep_100khz"]
    original_hits, filtered_hits, original_losses, filtered_losses = [], [], [], []
    try:
        for start in range(0, len(rows), 128):
            ix = rows[start:start + 128]
            iq = model_input(model, device_array(raw[ix, :, :model.waveform_samples], device))
            truth = device_array(labels[ix].astype(np.int64), device)
            for cutoff, correct, losses in ((original_cutoff, original_hits, original_losses),
                                             (filtered, filtered_hits, filtered_losses)):
                model.cutoff.copy_(cutoff)
                _, logits = model(iq)
                correct.append((logits.argmax(1) == truth).to(torch.uint8))
                losses.append(F.cross_entropy(logits, truth, reduction="none"))
    finally:
        model.cutoff.copy_(original_cutoff)
    original_ok = np.from_dlpack(torch.cat(original_hits)).astype(bool)
    filtered_ok = np.from_dlpack(torch.cat(filtered_hits)).astype(bool)
    return original_ok, filtered_ok, np.from_dlpack(torch.cat(original_losses)), np.from_dlpack(torch.cat(filtered_losses))


def audit(source, cache, checkpoint, seed, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    device = torch.device("cpu")
    model, cfg, updates = load_source_model(source, checkpoint, classes, device)
    if (updates != 846750 or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
            or cfg["source_roles"] != ["L", "V"] or model.grid_points != 256 or model.waveform_samples != 11000
            or model.reflect_boundary or Path(cfg["cache"]).resolve() != cache):
        raise ValueError("Expected the same fixed E30 source model as the band and time-distance audits")
    raw, labels, _ = roles["L" if smoke else "V"]
    rows = np.flatnonzero(labels >= 0)
    if smoke:
        rows = rows[:32]
    targets = labels[rows]
    predictions = measure_predictions(model, raw, labels, rows, device)

    # Metadata may only group completed, all-class predictions.
    with np.load(source / "L_index.npz", allow_pickle=False) as manifest:
        train_ticks = manifest["timestamp_ns"]
    if smoke:
        keep = np.ones(len(train_ticks), dtype=bool)
        keep[rows] = False
        gap = nearest_seconds(train_ticks[keep], labels[keep], train_ticks[rows], targets)
    else:
        with np.load(source / "V_index.npz", allow_pickle=False) as manifest:
            ticks = manifest["timestamp_ns"][rows]
        gap = nearest_seconds(train_ticks, roles["L"][1], ticks, targets)
    original_ok, masked_ok, original_ce, masked_ce = predictions
    if not np.isfinite(gap).all() or not np.isfinite(original_ce).all() or not np.isfinite(masked_ce).all():
        raise ValueError("Expected finite time distances and losses")
    strata = np.searchsorted(GAP_EDGES_SECONDS, gap, side="left")
    return dict(checkpoint=str(checkpoint), updates=updates, candidates=len(classes), smoke=smoke, seed=seed,
                edges_seconds=GAP_EDGES_SECONDS, all=describe_gap(np.ones(len(rows), dtype=bool), *predictions),
                strata=[describe_gap(strata == i, *predictions) for i in range(4)],
                matched_top1=within_beam_delta(targets, gap, original_ok.astype(int) - masked_ok, seed),
                matched_ce=within_beam_delta(targets, gap, masked_ce - original_ce, seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for attr in ("source", "cache", "checkpoint", "output"):
        parser.add_argument(attr, type=Path)
    parser.add_argument("--seed", type=int, default=2026111923)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.source.resolve(), args.cache.resolve(), args.checkpoint.resolve(), args.seed, args.smoke)
    write_json(args.output, result)
