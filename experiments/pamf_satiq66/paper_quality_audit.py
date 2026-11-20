"""Diagnose source-only single-packet beam accuracy by synchronization quality."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import SYMBOLS, write_json
from .paper_eval import model_input
from .paper_readout import load_source_model
from .paper_signal_readout import COMMON_SAMPLES, COMMON_SYMBOLS, SAMPLES_PER_SYMBOL
from .paper_train import source_inputs
from .train import device_array


def plateau_quality(iq):
    # Fixed early plateaus within the eight known, phase-synchronized symbols.
    plateaus = iq[..., :COMMON_SAMPLES].reshape(-1, 2, COMMON_SYMBOLS, SAMPLES_PER_SYMBOL)[..., :200]
    z = plateaus[:, 0].astype(np.float64) + 1j * plateaus[:, 1]
    signs = np.sign(SYMBOLS.real)
    dc = z.mean(axis=(1, 2))
    gain = (z * signs[None, :, None]).mean(axis=(1, 2))
    error = z - dc[:, None, None] - gain[:, None, None] * signs[None, :, None]
    residual = np.mean(np.abs(error) ** 2, axis=(1, 2))
    power = np.abs(gain) ** 2
    ratio = np.divide(residual, power, out=np.full_like(power, 1e12), where=power > 0)
    return -10 * np.log10(ratio.clip(1e-12, 1e12))


def matched_beam_contrast(labels, strata, correct, seed):
    groups = int(labels.max()) + 1
    low = strata == 0
    high = strata == 3
    count_low = np.bincount(labels[low], minlength=groups)
    count_high = np.bincount(labels[high], minlength=groups)
    hits_low = np.bincount(labels[low & correct], minlength=groups)
    hits_high = np.bincount(labels[high & correct], minlength=groups)
    pair = (count_low >= 2) & (count_high >= 2)
    if not pair.any():
        return None
    weight = np.minimum(count_low[pair], count_high[pair])
    rates_low = hits_low[pair] / count_low[pair]
    rates_high = hits_high[pair] / count_high[pair]
    delta = rates_high - rates_low
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(32):
        samples = rng.integers(len(weight), size=(64, len(weight)))
        estimates.append((delta * weight)[samples].sum(1) / weight[samples].sum(1))
    return dict(paired_beams=int(pair.sum()), matched_packets_per_end=int(weight.sum()),
                low_top1=float(np.average(rates_low, weights=weight)),
                high_top1=float(np.average(rates_high, weights=weight)),
                difference=float(np.average(delta, weights=weight)),
                beam_bootstrap_ci95=np.quantile(np.concatenate(estimates), [.025, .975]).tolist())


@torch.no_grad()
def audit(source, cache, checkpoint, seed, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    device = torch.device("cpu")
    model, cfg, updates = load_source_model(source, checkpoint, classes, device)
    if (cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256" or cfg["input_samples"] != 11000
            or cfg["source_roles"] != ["L", "V"] or Path(cfg["cache"]).resolve() != cache or updates != 564500):
        raise ValueError("Expected the fixed matched 11,000-point PAMF E20 source checkpoint")
    train_raw, train_labels, _ = roles["L"]
    raw, labels, _ = roles["V"]
    rng = np.random.default_rng(seed)
    fit = np.sort(rng.choice(len(train_raw), 32 if smoke else 8192, replace=False))
    quality_train = np.concatenate([plateau_quality(train_raw[fit[i:i + 128]]) for i in range(0, len(fit), 128)])
    edges = np.quantile(quality_train, [.25, .5, .75])
    if smoke:
        raw, labels = train_raw[fit], train_labels[fit]
    quality, targets, correct, sat_correct, ce = [], [], [], [], []
    satellite = classes[:, 0]
    for i in range(0, len(labels), 128):
        y = labels[i:i + 128]
        known = y >= 0
        if not known.any():
            continue
        y = y[known]
        iq = raw[i:i + 128][known]
        quality.append(plateau_quality(iq))
        _, logits = model(model_input(model, device_array(iq, device)))
        pred = np.from_dlpack(logits.argmax(dim=1))
        losses = F.cross_entropy(logits, device_array(y.astype(np.int64), device), reduction="none")
        ce.append(np.from_dlpack(losses))
        targets.append(y)
        correct.append(pred == y)
        sat_correct.append(satellite[pred] == satellite[y])
    quality, targets, correct, sat_correct, ce = map(np.concatenate, (quality, targets, correct, sat_correct, ce))
    if not np.isfinite(quality).all() or not np.isfinite(ce).all():
        raise ValueError("Nonfinite quality or CE in fixed source evaluation")
    strata = np.searchsorted(edges, quality, side="right")

    def describe(mask):
        return dict(records=int(mask.sum()), hits=int(correct[mask].sum()),
                    satellite_hits=int(sat_correct[mask].sum()), beams=int(np.unique(targets[mask]).size),
                    ce=float(ce[mask].mean(dtype=np.float64)) if mask.any() else None)

    return dict(checkpoint=str(checkpoint), updates=updates, seed=seed, fit_count=len(fit),
                fit_quality_quartiles=edges.tolist(), candidates=len(classes), smoke=smoke,
                all=describe(np.ones(len(targets), dtype=bool)),
                strata=[describe(strata == j) for j in range(4)],
                high_minus_low_within_beam=matched_beam_contrast(targets, strata, correct, seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "cache", "checkpoint", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.source.resolve(), args.cache.resolve(), args.checkpoint.resolve(), args.seed, args.smoke)
    write_json(args.output, result)
