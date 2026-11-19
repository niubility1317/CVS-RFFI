"""Measure CE and auxiliary gradient alignment on fixed native SatIQ source batches."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import write_json
from .paper_eval import normalize_packet
from .paper_native_pamf import native_classification_terms
from .paper_pamf import branch_prototype_loss
from .paper_readout import load_source_model
from .paper_train import SourceBatches, source_inputs
from .train import device_array


def objective_gradients(model, iq, target, satellites, config):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    _, logits, branch_logits, branches = model(iq, view_logits=True)
    ce, within, branch_ce, kl = native_classification_terms(logits, branch_logits, target, satellites)
    weights = config["pamf_sup"]
    losses = dict(
        within_satellite=weights["same_satellite_ce"] * within,
        branch_ce=weights["view_ce"] * branch_ce.mean(),
        branch_kl=weights["view_kl"] * kl,
        prototypes=config["branch_prototype_weight"] * branch_prototype_loss(branches),
    )
    terms = {"global_ce": ce, **losses, "all_auxiliary": sum(losses.values())}
    result = {}
    for index, (name, loss) in enumerate(terms.items()):
        gradients = torch.autograd.grad(loss, [p for _, p in parameters],
                                        retain_graph=index + 1 < len(terms), allow_unused=True)
        groups = {"encoder": [], "head": []}
        for (parameter_name, p), gradient in zip(parameters, gradients, strict=True):
            group = "head" if parameter_name == "head.weight" else "encoder"
            groups[group].append((gradient if gradient is not None else torch.zeros_like(p)).flatten())
        result[name] = {group: torch.cat(values) for group, values in groups.items()}
    return result


def alignment(main, auxiliary):
    dot = float(main @ auxiliary)
    main_squared = float(main @ main)
    auxiliary_squared = float(auxiliary @ auxiliary)
    if not np.isfinite([dot, main_squared, auxiliary_squared]).all() or main_squared <= 0:
        raise FloatingPointError("Global CE gradient is invalid")
    ratio = np.sqrt(auxiliary_squared / main_squared)
    return dict(norm_ratio=float(ratio), projection=dot / main_squared,
                cosine=dot / (main_squared * ratio) if ratio else None)


def summarize(batches):
    result = {}
    for group in batches[0]:
        result[group] = {}
        for term in batches[0][group]:
            records = [row[group][term] for row in batches]
            cosine = [r["cosine"] for r in records if r["cosine"] is not None]
            result[group][term] = dict(
                median_norm_ratio=float(np.median([r["norm_ratio"] for r in records])),
                median_projection=float(np.median([r["projection"] for r in records])),
                mean_cosine=float(np.mean(cosine)) if cosine else None,
                negative_batches=int(sum(value < 0 for value in cosine)),
                active_batches=len(cosine),
            )
    return result


def audit(source, cache, checkpoints, device, replay_seed=2026112007):
    classes, counts, roles = source_inputs(source, cache)
    raw, labels, original = roles["L"]
    satellites = device_array(classes[:, 0].astype(np.int64), device)
    batch_rows = None
    results = []
    for checkpoint in checkpoints:
        model, config, updates = load_source_model(source, checkpoint, classes, device)
        if (config.get("native_expansion") != "native160" or config["arm"] != "PAMF_SUP"
                or Path(config["cache"]).resolve() != cache.resolve()):
            raise ValueError("Expected a native160 PAMF-SUP checkpoint and its original source cache")
        if batch_rows is None:
            sampler = SourceBatches(original, counts, config["sampling_seed"])
            population = sampler.epoch()
            selected = np.sort(np.random.default_rng(replay_seed).choice(len(population), 64, replace=False))
            batch_rows = population[selected]
        model.train()
        torch.manual_seed(replay_seed)
        batches = []
        for indices in batch_rows:
            iq = normalize_packet(device_array(raw[indices, :, :model.waveform_samples], device))
            y = device_array(labels[indices].astype(np.int64), device)
            gradients = objective_gradients(model, iq, y, satellites, config)
            main = gradients.pop("global_ce")
            batches.append({group: {name: alignment(g, value[group]) for name, value in gradients.items()}
                            for group, g in main.items()})
        row = dict(checkpoint=str(checkpoint), updates=updates, summary=summarize(batches), batches=batches)
        results.append(row)
        print(updates, row["summary"], flush=True)
    return dict(replay_seed=replay_seed, source_role="L", batch_rows=batch_rows.tolist(), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "cache", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    write_json(args.output, audit(args.source, args.cache, args.checkpoint, torch.device(args.device)))
