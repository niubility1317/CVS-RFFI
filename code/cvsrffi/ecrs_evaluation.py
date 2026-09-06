"""Read-only, label-free three-path inference and source-only scoring."""
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .tensors import extract_meta_from_extra, unpack_batch

LEO_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _metadata(meta, keys, size):
    value = next((meta[k] for k in keys if k in meta), None)
    if value is None:
        return [None] * size
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value] * size
    if len(value) != size:
        raise ValueError("metadata length differs from batch: " + keys[0])
    return list(value)


def _stats(correct, total):
    return {"tx_correct": correct, "tx_total": total,
            "tx_acc": 100. * correct / total if total else None}


def evaluate_revision_paths(model, loader, device, scenario="clean", transform=None,
                            output_path=None, max_batches=0, source_metrics=True):
    """Predict without passing labels to the model; metrics use legal source V only.

    For truth-last prediction use source_metrics=False. No labels are read in that
    mode, and no truth is written. Explicit physical IDs are required for JSONL.
    Persistent model tensors and RNG are restored even after a failed evaluation.
    """
    if max_batches < 0:
        raise ValueError("max_batches must be nonnegative")
    modes = [(module, module.training) for module in model.modules()]
    state = {key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
             for key, value in model.state_dict().items()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    counts = defaultdict(lambda: [0, 0])
    groups = defaultdict(lambda: {"per_rx": defaultdict(lambda: [0, 0]),
                                  "per_day": defaultdict(lambda: [0, 0]),
                                  "per_rx_day": defaultdict(lambda: [0, 0])})
    rescue = harm = total = 0
    output = None
    seen_ids = set()
    fusion_active = bool(getattr(model, "_ecrs_fusion_head_initialized", True))
    try:
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            output = path.open("x", encoding="utf-8", newline="\n")
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                x, labels, extra = unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                if transform is not None:
                    x = transform(x, batch_idx)
                result = model.forward_identity(x)
                raw = result.get("tx_logits_raw_infer", result.get("tx_logits_raw", result.get("tx_logits")))
                if raw is None:
                    raise ValueError("identity inference must return raw logits")
                logits = {"raw": raw}
                if result.get("resp_tx_logits") is not None:
                    logits["response"] = result["resp_tx_logits"]
                if "response" in logits or result.get("tx_logits_fused") is not None:
                    logits["fused"] = (result.get("tx_logits_fused", result.get("tx_logits", raw))
                                       if fusion_active else raw)
                if result.get("z_id_fused") is not None and hasattr(model, "_classify_identity_feature"):
                    logits["common_head"] = (model._classify_identity_feature(result["z_id_fused"], None)
                                               if fusion_active else raw)
                size = len(x)
                for name, tensor in logits.items():
                    if tensor.ndim != 2 or tensor.shape[0] != size or not torch.isfinite(tensor).all():
                        raise ValueError("invalid or nonfinite " + name + " logits")
                cpu_logits = {name: value.detach().cpu() for name, value in logits.items()}
                predictions = {name: value.argmax(1).tolist() for name, value in cpu_logits.items()}
                truth = torch.as_tensor(labels).detach().cpu().reshape(-1).tolist() if source_metrics else None
                if source_metrics and len(truth) != size:
                    raise ValueError("label count differs from input")
                meta = extract_meta_from_extra(extra) or {}
                ids = _metadata(meta, ("physical_sample_id",), size)
                receivers = _metadata(meta, ("receiver_id", "rx_i"), size)
                days = _metadata(meta, ("day_id", "day_i"), size)
                for i in range(size):
                    sample_id = ids[i]
                    if output is not None:
                        if sample_id is None:
                            raise ValueError("prediction output requires physical_sample_id")
                        sample_id = str(sample_id)
                        if sample_id in seen_ids:
                            raise ValueError("duplicate physical_sample_id in scenario: " + sample_id)
                        seen_ids.add(sample_id)
                    valid = source_metrics and 0 <= int(truth[i]) < raw.shape[1]
                    if valid:
                        correct = {name: int(pred[i] == int(truth[i])) for name, pred in predictions.items()}
                        for name, hit in correct.items():
                            counts[name][0] += hit
                            counts[name][1] += 1
                            for key, group in (("per_rx", receivers[i]), ("per_day", days[i]),
                                               ("per_rx_day", f"{receivers[i]}:{days[i]}" if receivers[i] is not None and days[i] is not None else None)):
                                if group is not None:
                                    cell = groups[name][key][str(group)]
                                    cell[0] += hit
                                    cell[1] += 1
                        if "fused" in correct:
                            rescue += int(not correct["raw"] and correct["fused"])
                            harm += int(correct["raw"] and not correct["fused"])
                    if output is not None:
                        record = {"physical_sample_id": sample_id, "receiver_id": receivers[i],
                                  "day_id": days[i], "scenario": scenario,
                                  "predictions": {name: pred[i] for name, pred in predictions.items()},
                                  "logits": {name: tensor[i].tolist() for name, tensor in cpu_logits.items()}}
                        if source_metrics:
                            record["source_truth"] = int(truth[i])
                        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    total += 1
                if output is not None:
                    output.flush()
    finally:
        if output is not None:
            output.close()
        model.load_state_dict(state, strict=True)
        for module, training in modes:
            module.training = training
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    paths = {}
    for name, (correct, count) in counts.items():
        entry = _stats(correct, count)
        for key, cells in groups[name].items():
            entry[key] = {group: _stats(*pair) for group, pair in cells.items()}
        entry["rx_floor"] = min((v["tx_acc"] for v in entry["per_rx"].values()), default=None)
        entry["day_floor"] = min((v["tx_acc"] for v in entry["per_day"].values()), default=None)
        paths[name] = entry
    scored = counts["raw"][1]
    return {"scenario": scenario, "paths": paths, "count": total, "source_metrics": source_metrics,
            "rescue": rescue if source_metrics else None, "harm": harm if source_metrics else None,
            "net": rescue - harm if source_metrics else None,
            "net_pp": 100. * (rescue - harm) / scored if scored else None,
            "fusion_active": fusion_active, "output_path": str(output_path) if output_path else None}


def choose_source_score(paths_by_scenario, declared_path="raw"):
    """Return source three-LEO mean or None; target and independent-B0 tests are external.

    All declared scenes and receiver cells must exist. The caller fixes the path
    before training; this helper never chooses the best of raw/response/fused.
    """
    if declared_path not in ("raw", "fused"):
        raise ValueError("declared selection path must be raw or fused")
    scenes = ("clean",) + LEO_SCENARIOS
    try:
        selected = [paths_by_scenario[s]["paths"][declared_path] for s in scenes]
        raw = [paths_by_scenario[s]["paths"]["raw"] for s in scenes]
        if any(paths_by_scenario[s].get("source_metrics") is not True for s in scenes):
            return None
        scores = [float(p["tx_acc"]) for p in selected + raw]
        if not all(math.isfinite(v) for v in scores):
            return None
        score = sum(float(p["tx_acc"]) for p in selected[1:]) / 3.
        if declared_path == "raw":
            return score
        if any(not paths_by_scenario[s].get("fusion_active", False) for s in scenes):
            return None
        if selected[0]["tx_acc"] < raw[0]["tx_acc"] - .5:
            return None
        for candidate, control in zip(selected, raw):
            if not control["per_rx"] or candidate["per_rx"].keys() != control["per_rx"].keys():
                return None
            for rx in control["per_rx"]:
                c, r = float(candidate["per_rx"][rx]["tx_acc"]), float(control["per_rx"][rx]["tx_acc"])
                if not math.isfinite(c) or not math.isfinite(r) or c < r - 1.:
                    return None
        if score < sum(float(p["tx_acc"]) for p in raw[1:]) / 3. + 1.:
            return None
        if sum(paths_by_scenario[s]["net"] for s in LEO_SCENARIOS) <= 0:
            return None
        return score
    except (KeyError, TypeError, ValueError):
        return None
