"""Calibrate a single-packet blend of the native PAMF and nonlinear logits."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import write_json
from .paper_direct_branches import blend_logits, model_scores
from .paper_native_pamf import NATIVE_PATHS
from .paper_readout import load_source_model
from .paper_sup_prototype_readout import enroll_rows
from .paper_train import source_inputs


ALPHAS = (.10, .20, .35, .50)
QUANTILES = {"low25": .25, "low50": .50, "low75": .75}
NONLINEAR = NATIVE_PATHS.index("nonlinear")


def gate_scores(fused):
    top_scores = fused.topk(2, dim=1).values
    return fused.argmax(1), top_scores[:, 0] - top_scores[:, 1]


def gate_active(margin, name, limits):
    return torch.ones_like(margin, dtype=torch.bool) if name == "all" else margin <= limits[name]


def fit_gates(batches):
    margins, outcomes = [], []
    for fused, nonlinear, truth in batches:
        base, margin = gate_scores(fused)
        predictions = [base, *(blend_logits(fused, nonlinear, a).argmax(1) for a in ALPHAS)]
        margins.append(margin)
        outcomes.append(torch.stack([p == truth for p in predictions]))
    margin = torch.cat(margins)
    correct = torch.cat(outcomes, dim=1)
    limits = {k: float(torch.quantile(margin, q)) for k, q in QUANTILES.items()}
    hits = {("all", 0.): int(correct[0].sum())}
    for name in ("all", *QUANTILES):
        active = gate_active(margin, name, limits)
        for alpha, candidate in zip(ALPHAS, correct[1:], strict=True):
            hits[name, alpha] = int(torch.where(active, candidate, correct[0]).sum())
    return max(hits, key=hits.get), hits, limits


def score_gates(batches, chosen, limits, always=None):
    rules = {"calibrated": chosen}
    if always is not None:
        rules["always"] = always
    stats = {name: dict(correct=0, gained_vs_fused=0, lost_vs_fused=0, active=0)
             for name in ("fused", "nonlinear", *rules)}
    either_correct = 0
    records = 0
    for fused, nonlinear, truth in batches:
        pred, margin = gate_scores(fused)
        base_ok = pred == truth
        nonlinear_ok = nonlinear.argmax(1) == truth
        scores = {"fused": (base_ok, torch.zeros_like(base_ok)),
                  "nonlinear": (nonlinear_ok, torch.ones_like(base_ok))}
        records += len(truth)
        either_correct += int((base_ok | nonlinear_ok).sum())
        for name, (gate, alpha) in rules.items():
            active = gate_active(margin, gate, limits) if alpha else torch.zeros_like(base_ok)
            blended = blend_logits(fused[active], nonlinear[active], alpha).argmax(1)
            correct = base_ok.clone()
            correct[active] = blended == truth[active]
            scores[name] = (correct, active)
        for name, (correct, active) in scores.items():
            stats[name]["correct"] += int(correct.sum())
            stats[name]["gained_vs_fused"] += int((correct & ~base_ok).sum())
            stats[name]["lost_vs_fused"] += int((~correct & base_ok).sum())
            stats[name]["active"] += int(active.sum())
    return dict(records=records, heads=stats, either_native_path_correct=either_correct)


def nonlinear_scores(model, raw, labels, rows, device):
    for fused, branches, y in model_scores(model, raw, labels, rows, device):
        yield fused, branches[NONLINEAR], y


@torch.no_grad()
def compare(source, cache, checkpoint, seed, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    device = torch.device("cpu")
    model, cfg, steps = load_source_model(source, checkpoint, classes, device)
    if (steps != 846750 or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
            or cfg["input_samples"] != 11000 or cfg["source_roles"] != ["L", "V"]
            or Path(cfg["cache"]).resolve() != cache):
        raise ValueError("Expected the exact same-header native PAMF E30 source checkpoint")
    raw_l, labels_l, _ = roles["L"]
    calibration = (np.flatnonzero(labels_l >= 0)[:32] if smoke
                   else enroll_rows(labels_l, len(classes), 8, seed))
    chosen, hits, limits = fit_gates(nonlinear_scores(model, raw_l, labels_l, calibration, device))
    always = max((r for r in hits if r[0] == "all"), key=hits.get)
    print(f"Source L {len(calibration)} packets: selected {chosen}, always-on {always}", flush=True)
    raw, labels, _ = roles["L" if smoke else "V"]
    query = np.flatnonzero(labels >= 0)
    if smoke:
        query = query[32:64]
    result = score_gates(nonlinear_scores(model, raw, labels, query, device), chosen, limits, always)
    return dict(checkpoint=str(checkpoint), updates=steps, candidates=len(classes), smoke=smoke,
                calibration_seed=seed, calibration_packets=len(calibration), chosen=chosen, always=always,
                calibration_hits={f"{name}/{a}": n for (name, a), n in hits.items()}, margin_limits=limits,
                **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "cache", "checkpoint", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = compare(args.source.resolve(), args.cache.resolve(), args.checkpoint.resolve(), args.seed, args.smoke)
    write_json(args.output, result)
