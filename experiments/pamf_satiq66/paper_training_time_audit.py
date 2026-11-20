"""Compare two matched frozen PAMF source-training choices on the same packets."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_band_time_audit import within_beam_delta
from .paper_eval import model_input
from .paper_grid_paired_audit import paired_delta_interval
from .paper_readout import load_source_model
from .paper_time_gap_audit import GAP_EDGES_SECONDS, nearest_seconds
from .paper_train import source_inputs
from .train import device_array


OPTIONS = {"warmup": ("native_logit_scale_warmup", None),
           "pa-highband": ("native_pa_highband_gain", 1.),
           "shared-satellite-head": ("native_shared_satellite_head", False),
           "prototype-positive-gap": ("native_prototype_positives", "random_two")}


def validate_pair(baseline, candidate, comparison):
    option, default = OPTIONS[comparison]
    ignored = {option, "resume"}
    if ({k: v for k, v in baseline.items() if k not in ignored}
            != {k: v for k, v in candidate.items() if k not in ignored}):
        raise ValueError(f"Frozen configs differ beyond {option} and their resume paths")
    if baseline.get(option, default) != default or candidate.get(option, default) == default:
        raise ValueError(f"Expected default and active values of {option} in that order")
    if comparison == "prototype-positive-gap" and (baseline.get(option) != "random_two"
                                                     or candidate[option] != "farthest_two"):
        raise ValueError("Expected exact random-two and farthest-two prototype policies")
    return candidate[option]


def summarize(mask, labels, classes, baseline_ok, candidate_ok, baseline_ce, candidate_ce, seed):
    base, candidate = baseline_ok[mask], candidate_ok[mask]
    outcome = dict(records=int(mask.sum()), baseline_hits=int(base.sum()), candidate_hits=int(candidate.sum()),
                   gained=int((candidate & ~base).sum()), lost=int((base & ~candidate).sum()))
    if not mask.any():
        return outcome
    target = labels[mask]
    delta = candidate.astype(int) - base
    _, sat = np.unique(classes[target, 0], return_inverse=True)
    outcome.update(baseline_ce=float(baseline_ce[mask].mean(dtype=np.float64)),
                   candidate_ce=float(candidate_ce[mask].mean(dtype=np.float64)),
                   ce_diff=float((candidate_ce[mask] - baseline_ce[mask]).mean(dtype=np.float64)),
                   satellites=int(sat.max()) + 1,
                   satellite_interval95=paired_delta_interval(np.bincount(sat), np.bincount(sat, weights=delta), seed))
    return outcome


@torch.no_grad()
def measure(model, raw, labels, rows, device):
    hits, losses = [], []
    for start in range(0, len(rows), 128):
        ix = rows[start:start + 128]
        truth = device_array(labels[ix].astype(np.int64), device)
        iq = model_input(model, device_array(raw[ix, :, :model.waveform_samples], device))
        _, logits = model(iq)
        hits.append((logits.argmax(1) == truth).to(torch.uint8))
        losses.append(F.cross_entropy(logits, truth, reduction="none"))
    return np.from_dlpack(torch.cat(hits)).astype(bool), np.from_dlpack(torch.cat(losses))


def audit(source, cache, checkpoints, comparison, epoch, smoke=False):
    device = torch.device("cpu")
    classes, _, roles = source_inputs(source, cache)
    models, configs = [], []
    expected = epoch * 28225
    for checkpoint in checkpoints:
        model, cfg, updates = load_source_model(source, checkpoint, classes, device)
        if (updates != expected or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
                or cfg["source_roles"] != ["L", "V"] or cfg["augmentation"] != "none" or cfg["epochs"] != epoch
                or model.grid_points != 256 or model.waveform_samples != 11000 or model.reflect_boundary
                or not cfg["align_reference_time"] or Path(cfg["cache"]).resolve() != cache or len(classes) != 3103):
            raise ValueError(f"Expected an exact source-only PAMF checkpoint at fixed epoch {epoch}")
        models.append(model)
        configs.append(cfg)
    active = validate_pair(*configs, comparison)
    raw, labels, _ = roles["L" if smoke else "V"]
    rows = np.flatnonzero(labels >= 0)
    if smoke:
        rows = rows[:32]
    base, candidate = [measure(model, raw, labels, rows, device) for model in models]
    if not (np.isfinite(base[1]).all() and np.isfinite(candidate[1]).all()):
        raise ValueError("Frozen model produced nonfinite CE")

    # Time metadata is used only to group completed predictions.
    targets = labels[rows]
    with np.load(source / "L_index.npz", allow_pickle=False) as f:
        train = f["timestamp_ns"]
    if smoke:
        reference = np.ones(len(train), dtype=bool)
        reference[rows] = False
        gap = nearest_seconds(train[reference], labels[reference], train[rows], targets)
    else:
        with np.load(source / "V_index.npz", allow_pickle=False) as f:
            ticks = f["timestamp_ns"][rows]
        gap = nearest_seconds(train, roles["L"][1], ticks, targets)
    bins = np.searchsorted(GAP_EDGES_SECONDS, gap, side="left")
    seed = configs[0]["evaluation_seed"]
    items = (targets, classes, base[0], candidate[0], base[1], candidate[1], seed)
    return dict(comparison=comparison, active_option=active, updates=expected, candidates=len(classes), smoke=smoke,
                epoch=epoch, checkpoints=list(map(str, checkpoints)), edges_seconds=GAP_EDGES_SECONDS, seed=seed,
                overall=summarize(np.ones(len(rows), dtype=bool), *items),
                by_gap=[summarize(bins == i, *items) for i in range(4)],
                far=summarize(gap > 30, *items),
                matched_effect=within_beam_delta(targets, gap, candidate[0].astype(int) - base[0], seed),
                matched_ce=within_beam_delta(targets, gap, candidate[1] - base[1], seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for attr in ("source", "cache", "baseline", "candidate"):
        parser.add_argument(attr, type=Path)
    parser.add_argument("comparison", choices=tuple(OPTIONS))
    parser.add_argument("epoch", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, audit(args.source.resolve(), args.cache.resolve(),
                                 (args.baseline.resolve(), args.candidate.resolve()),
                                 args.comparison, args.epoch, args.smoke))
