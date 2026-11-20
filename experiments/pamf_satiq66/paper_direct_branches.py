"""Measure all-beam accuracy from the five existing PAMF classification paths."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import write_json
from .paper_eval import model_input
from .paper_native_pamf import NATIVE_PATHS
from .paper_readout import load_source_model
from .paper_sup_prototype_readout import enroll_rows
from .paper_train import source_inputs
from .train import device_array


def blend_logits(native, branch_mean, alpha):
    if alpha == 0:
        return native
    return torch.lerp(native, branch_mean, alpha)


def select_alpha(batches):
    hits = dict.fromkeys((0., .10, .20, .35, .50), 0)
    for fused, mean, truth in batches:
        for alpha in hits:
            hits[alpha] += int((blend_logits(fused, mean, alpha).argmax(dim=1) == truth).sum())
    return max(hits, key=hits.get), hits


def model_scores(model, raw, labels, rows, device):
    for start in range(0, len(rows), 128):
        ix = rows[start:start + 128]
        iq = model_input(model, device_array(raw[ix, :, :model.waveform_samples], device))
        y = device_array(labels[ix].astype(np.int64), device)
        _, native, branches, _ = model(iq, view_logits=True)
        yield native, branches, y


def compare_scores(batches, alpha):
    names = ("fused", *NATIVE_PATHS, "branch_mean", "calibrated_mix")
    metrics = {name: dict(correct=0, gained_vs_fused=0, lost_vs_fused=0) for name in names}
    any_path_correct = 0
    num_packets = 0
    for native, branches, y in batches:
        branch_mean = branches.mean(dim=0)
        mixed = blend_logits(native, branch_mean, alpha)
        z = (native, *branches.unbind(0), branch_mean, mixed)
        fused_ok = native.argmax(dim=1) == y
        any_path_correct += int(((branches.argmax(dim=-1) == y[None]).any(dim=0) | fused_ok).sum())
        num_packets += len(y)
        for name, logits in zip(names, z, strict=True):
            correct = logits.argmax(dim=1) == y
            row = metrics[name]
            row["correct"] += int(correct.sum())
            row["gained_vs_fused"] += int((correct & ~fused_ok).sum())
            row["lost_vs_fused"] += int((~correct & fused_ok).sum())
    return dict(records=num_packets, heads=metrics, any_native_path_correct=any_path_correct)


@torch.no_grad()
def compare(source, cache, checkpoint, seed, smoke=False):
    classes, _, roles = source_inputs(source, cache)
    cpu = torch.device("cpu")
    model, cfg, updates = load_source_model(source, checkpoint, classes, cpu)
    if (updates != 846750 or cfg["arm"] != "PAMF_SUP" or cfg["native_expansion"] != "fused256"
            or cfg["input_samples"] != 11000 or cfg["source_roles"] != ["L", "V"]
            or Path(cfg["cache"]).resolve() != cache):
        raise ValueError("Expected the exact same-header native PAMF E30 source checkpoint")
    raw_l, labels_l, _ = roles["L"]
    calibration = (np.flatnonzero(labels_l >= 0)[:32] if smoke
                   else enroll_rows(labels_l, len(classes), 8, seed))
    samples = model_scores(model, raw_l, labels_l, calibration, cpu)
    alpha, hits = select_alpha((native, branches.mean(dim=0), y) for native, branches, y in samples)
    print(f"Selected branch mixture {alpha} on {len(calibration)} source L packets", flush=True)
    raw, labels, _ = roles["L" if smoke else "V"]
    query = np.flatnonzero(labels >= 0)
    if smoke:
        query = query[32:64]
    outcomes = compare_scores(model_scores(model, raw, labels, query, cpu), alpha)
    return dict(checkpoint=str(checkpoint), updates=updates, candidates=len(classes), smoke=smoke,
                calibration_seed=seed, calibration_packets=len(calibration), calibration_hits=hits,
                branch_mix=alpha, **outcomes)


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
