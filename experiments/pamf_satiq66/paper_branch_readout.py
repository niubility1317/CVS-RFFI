"""Read the five original PAMF-SUP branches on frozen source checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import write_json
from .paper_calibrated_readout import evaluate_panels, feature_geometry, fit_transforms
from .paper_eval import normalize_packet
from .paper_native_pamf import NATIVE_PATHS, PaperNativePAMF
from .paper_readout import load_source_model
from .paper_signal_readout import panel_rows, source_panels
from .paper_train import source_inputs
from .train import device_array


@torch.no_grad()
def branch_vectors(model, raw, rows, device):
    blocks = []
    for start in range(0, len(rows), 128):
        ix = rows[start:start + 128]
        iq = normalize_packet(device_array(raw[ix, :, :model.waveform_samples], device))
        fused, _, _, branches = model(iq, view_logits=True)
        blocks.append(tuple(F.normalize(x, dim=1).cpu() for x in (fused, *branches.unbind(0))))
    return dict(zip(("fused", *NATIVE_PATHS),
                    (torch.cat(part) for part in zip(*blocks, strict=True)), strict=True))


def compare(source, cache, checkpoints, seed, fit_count, device):
    classes, _, roles = source_inputs(source, cache)
    l = roles["L"][0]
    v, labels, _ = roles["V"]
    panels, cutoff = source_panels(source, labels, classes, seed)
    needed, panels = panel_rows(panels, len(v))
    fit = np.sort(np.random.default_rng(seed).choice(len(l), fit_count, replace=False))
    results = []
    for checkpoint in checkpoints:
        model, config, updates = load_source_model(source, checkpoint, classes, device)
        if (not isinstance(model, PaperNativePAMF) or config["arm"] != "PAMF_SUP"
                or Path(config["cache"]).resolve() != cache.resolve()):
            raise ValueError("Branch diagnosis requires the native PAMF-SUP and its original source cache")
        known = branch_vectors(model, l, fit, device)
        query = branch_vectors(model, v, needed, device)
        metrics = {}
        for name, vectors in query.items():
            raw, _, white = fit_transforms(known[name], vectors)
            metrics[name] = dict(geometry=feature_geometry(raw), raw=evaluate_panels(raw, panels, device),
                                 partial_whiten=evaluate_panels(white, panels, device))
        results.append(dict(checkpoint=str(checkpoint), updates=updates, paths=metrics))
        print(f"Scored {checkpoint.parent.parent.name} step {updates}", flush=True)
    return dict(source_train_fit_count=fit_count, source_fit_seed=seed, source_eval_packets=len(needed),
                chronological_cutoff_utc=cutoff, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=2026111907)
    parser.add_argument("--fit-count", type=int, default=8192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, compare(args.source, args.cache, args.checkpoint, args.seed,
                                   args.fit_count, torch.device(args.device)))
