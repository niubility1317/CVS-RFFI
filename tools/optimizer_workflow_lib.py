#!/usr/bin/env python
"""Shared helpers for CV-SincNet optimizer workflow tooling."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


METRIC_KEYS = (
    "strict_udu",
    "sat_mean",
    "sat_floor",
    "receiver_floor",
    "joint_score",
    "risk_adjusted",
)

MECHANISM_FIELDS = {
    "fishr_active_after100": ("fishr_active_after100",),
    "fishr_domain_count_after100": ("fishr_domain_count_after100",),
    "style_ready_after100": ("style_ready_after100",),
    "style_batch_active": ("style_batch_active", "diag_style_batch_active"),
    "global_style_enabled": ("global_style_summary.enabled",),
    "cgrl_enabled": ("global_fed_cgrl_summary.enabled",),
    "fed_proto_loss_nonzero": ("fed_proto_loss_nonzero", "fed_proto_active"),
    "fed_coral_loss_nonzero": ("fed_coral_loss_nonzero", "fed_coral_active"),
    "fed_fishr_enabled": ("global_fed_fishr_summary.enabled", "global_fed_fishr_enabled"),
    "fed_fishr_active": ("global_fed_fishr_summary.active", "global_fed_fishr_active"),
    "fed_fishr_reweight_active": ("global_fed_fishr_summary.reweight_active", "global_fed_fishr_reweight_active"),
    "fed_fishr_active_classes": ("global_fed_fishr_summary.active_classes", "global_fed_fishr_active_classes"),
    "fed_fishr_mismatch_mean": ("global_fed_fishr_summary.mismatch_mean", "global_fed_fishr_mismatch_mean"),
    "fed_fishr_weight_max_delta": ("global_fed_fishr_summary.weight_max_delta", "global_fed_fishr_weight_max_delta"),
}


def read_text_compat(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_json_compat(path: Path) -> Any:
    return json.loads(read_text_compat(path))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def nested_get(value: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def first_present(value: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if "." in key:
            candidate = nested_get(value, key)
        else:
            candidate = value.get(key)
        if candidate not in (None, ""):
            return candidate
    return None


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def infer_batch(path: Path, root: Any) -> str:
    if isinstance(root, Mapping) and root.get("batch"):
        return str(root["batch"])
    candidates = [path.name, path.parent.name, path.parent.parent.name]
    for text in candidates:
        match = re.search(r"(CEN\d+[A-Z]*|FED\d+[A-Z]*)", text, re.I)
        if match:
            return match.group(1).upper()
    return path.parent.parent.name


def infer_lane(path: Path, root: Any) -> str:
    if isinstance(root, Mapping) and root.get("lane"):
        return str(root["lane"])
    text = " ".join([path.name, path.parent.name, path.parent.parent.name]).lower()
    if "fed" in text or "federated" in text or "vmb" in text:
        return "federated_vmb"
    return "centralized"


def item_list(root: Any) -> List[Mapping[str, Any]]:
    if isinstance(root, list):
        return [item for item in root if isinstance(item, Mapping)]
    if not isinstance(root, Mapping):
        return []
    for key in ("effective_items", "items", "candidates", "records", "results", "commands"):
        value = root.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def satellite_metrics(value: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    sat = (
        value.get("sat")
        or value.get("satellite")
        or value.get("sat_by_scenario")
        or value.get("satellites")
        or {}
    )
    if not isinstance(sat, Mapping) or not sat:
        return {"sat_mean": as_float(value.get("sat_mean")), "sat_floor": as_float(value.get("sat_floor"))}
    vals = [as_float(v) for v in sat.values()]
    vals = [v for v in vals if v is not None]
    return {
        "sat_mean": as_float(value.get("sat_mean")) if value.get("sat_mean") is not None else (sum(vals) / len(vals) if vals else None),
        "sat_floor": as_float(value.get("sat_floor")) if value.get("sat_floor") is not None else (min(vals) if vals else None),
    }


def normalize_metric(value: Any) -> Dict[str, Optional[float]]:
    if not isinstance(value, Mapping):
        return {key: None for key in METRIC_KEYS}
    nested = value.get("metrics")
    if isinstance(nested, Mapping):
        merged: Dict[str, Any] = dict(nested)
        merged.update({k: v for k, v in value.items() if k not in merged})
        value = merged
    sat = satellite_metrics(value)
    splits = value.get("splits") if isinstance(value.get("splits"), Mapping) else {}
    strict = first_present(value, ("strict_udu", "test_unseen_day_unseen_rx", "strict_udu_acc"))
    if strict is None:
        strict = splits.get("unseen_day_unseen_rx")
    receiver = first_present(value, ("receiver_floor", "rx8_udu", "worst_rx"))
    return {
        "strict_udu": as_float(strict),
        "sat_mean": sat["sat_mean"],
        "sat_floor": sat["sat_floor"],
        "receiver_floor": as_float(receiver),
        "joint_score": as_float(first_present(value, ("joint_score",))),
        "risk_adjusted": as_float(first_present(value, ("risk_adjusted", "risk_adjusted_score"))),
    }


def best_blocks(root: Any) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(root, Mapping):
        return {}
    out: Dict[str, Mapping[str, Any]] = {}
    if isinstance(root.get("batch_bests"), Mapping):
        out.update({str(k): v for k, v in root["batch_bests"].items() if isinstance(v, Mapping)})
    if isinstance(root.get("batch_best"), Mapping):
        out.update({str(k): v for k, v in root["batch_best"].items() if isinstance(v, Mapping)})
    mapping = {
        "batch_best_clean": "clean",
        "batch_best_satellite": "satellite",
        "batch_best_joint": "joint",
        "batch_best_risk_adjusted": "risk_adjusted",
        "batch_best_final_clean": "final_clean",
        "batch_best_final_satellite": "final_satellite",
    }
    for source, label in mapping.items():
        if isinstance(root.get(source), Mapping):
            out[label] = root[source]
    return out


def candidate_id(item: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "candidate", "experiment_id", "id"):
        value = item.get(key)
        if value:
            return str(value)
    run_name = item.get("run_name") or item.get("name") or item.get("run")
    if run_name:
        match = re.match(r"((?:CEN|FED)\d+[A-Z]*_[RA]\d+)", str(run_name), re.I)
        if match:
            return match.group(1).upper()
    return "UNKNOWN"


def run_name(item: Mapping[str, Any]) -> Optional[str]:
    value = first_present(item, ("run_name", "experiment_id", "name", "run"))
    return str(value) if value is not None else None


def status_for_item(item: Mapping[str, Any]) -> str:
    if item.get("status"):
        return str(item["status"])
    if item.get("collapse_flag") is True:
        return "collapsed"
    if item.get("finished") is True:
        return "finished"
    if item.get("startup_failed") is True:
        return "startup_failed"
    return "unknown"


def collapse_flags(item: Mapping[str, Any]) -> List[str]:
    flags: List[str] = []
    raw = item.get("collapse_flags") or item.get("scenario_collapse_signals") or []
    if isinstance(raw, list):
        flags.extend(str(v) for v in raw)
    elif raw:
        flags.append(str(raw))
    if item.get("collapse_flag") is True:
        flags.append("collapse_flag_true")
    return sorted(set(flags))


def mechanism_activation(item: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for target, keys in MECHANISM_FIELDS.items():
        out[target] = first_present(item, keys)
    return out


def item_views(item: Mapping[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if isinstance(item.get("modes"), Mapping):
        for mode, metric in item["modes"].items():
            out[str(mode)] = normalize_metric(metric)
    if isinstance(item.get("best_modes"), Mapping):
        for mode, metric in item["best_modes"].items():
            out[f"best_{mode}"] = normalize_metric(metric)
    for key, label in (
        ("best_clean", "clean"),
        ("best_satellite", "satellite"),
        ("best_joint", "joint"),
        ("best_risk_adjusted", "risk_adjusted"),
        ("best_mode_by_strict", "clean"),
        ("final", "final"),
    ):
        if isinstance(item.get(key), Mapping):
            out[label] = normalize_metric(item[key])
    direct = normalize_metric(item)
    if any(value is not None for value in direct.values()):
        out.setdefault("direct", direct)
    return out


def standardize_summary(root: Any, source_path: Path) -> Dict[str, Any]:
    batch = infer_batch(source_path, root)
    lane = infer_lane(source_path, root)
    items = []
    for item in item_list(root):
        items.append(
            {
                "candidate_id": candidate_id(item),
                "run_name": run_name(item),
                "status": status_for_item(item),
                "views": item_views(item),
                "collapse_flags": collapse_flags(item),
                "mechanism_activation": mechanism_activation(item),
                "source_paths": {
                    "log_path": item.get("log_path") or item.get("out_path"),
                    "run_path": item.get("run_path"),
                    "config_path": item.get("config_path"),
                },
            }
        )
    batch_bests: Dict[str, Any] = {}
    for label, block in best_blocks(root).items():
        batch_bests[label] = {
            "candidate_id": candidate_id(block),
            "run_name": run_name(block),
            "mode": block.get("mode"),
            "metrics": normalize_metric(block),
        }
    if isinstance(root, Mapping):
        evidence_hash = root.get("evidence_hash")
        hard_error_count = root.get("hard_error_count")
        finished_count = root.get("finished_count")
    else:
        evidence_hash = hard_error_count = finished_count = None
    return {
        "schema": "optimizer_batch_summary_v1",
        "batch": batch,
        "lane": lane,
        "status": "completed" if finished_count else "unknown",
        "source_path": str(source_path),
        "evidence_hash": evidence_hash,
        "finished_count": finished_count,
        "hard_error_count": hard_error_count,
        "candidate_count": len(items),
        "items": items,
        "batch_bests": batch_bests,
    }


def load_standard_summary(path: Path) -> Dict[str, Any]:
    root = load_json_compat(path)
    if isinstance(root, Mapping) and root.get("schema") == "optimizer_batch_summary_v1":
        return dict(root)
    return standardize_summary(root, path)


def candidate_best_metrics(item: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    views = item.get("views") if isinstance(item.get("views"), Mapping) else {}
    metrics = [v for v in views.values() if isinstance(v, Mapping)]
    if not metrics:
        return {key: None for key in METRIC_KEYS}
    comparable = [
        metric
        for metric in metrics
        if as_float(metric.get("strict_udu")) is not None
        and as_float(metric.get("sat_floor")) is not None
        and as_float(metric.get("receiver_floor")) is not None
    ]
    if comparable:
        metrics = comparable

    def score(metric: Mapping[str, Any]) -> float:
        vals = [
            as_float(metric.get("risk_adjusted")),
            as_float(metric.get("joint_score")),
            as_float(metric.get("strict_udu")),
            as_float(metric.get("sat_floor")),
            as_float(metric.get("receiver_floor")),
        ]
        return max([v for v in vals if v is not None], default=-1e9)

    best = max(metrics, key=score)
    return {key: as_float(best.get(key)) for key in METRIC_KEYS}


def dominates(left: Mapping[str, Optional[float]], right: Mapping[str, Optional[float]], keys: Iterable[str]) -> bool:
    left_vals = [left.get(k) for k in keys]
    right_vals = [right.get(k) for k in keys]
    if any(v is None for v in left_vals) or any(v is None for v in right_vals):
        return False
    return all(float(a) >= float(b) for a, b in zip(left_vals, right_vals)) and any(
        float(a) > float(b) for a, b in zip(left_vals, right_vals)
    )
