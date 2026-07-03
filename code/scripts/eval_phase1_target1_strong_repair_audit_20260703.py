#!/usr/bin/env python
"""Strong target-1 audit for Phase1 LEO feature repair.

This script evaluates whether source-only LEO repair is strong enough to count
as an anti-LEO correction, not merely a feature-pair alignment improvement.
It audits closed old-class recovery, scenario and TX floors, feature-space
margin/compactness, and unknown safety against the identity repaired feature
baseline. Target labels are used only for final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_VARIANTS = (
    "LEOADAPT3_IDENTITY",
    "LEOADAPT3_MEANSHIFT",
    "LEOADAPT3_NORMSHIFT",
    "LEOADAPT3_LINR_COS",
    "LEOADAPT3_MLP_ID",
)
SOURCE_TX_IDS = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")


def canonical_tx_id(value: object) -> str:
    text = str(value)
    if text.startswith("tx"):
        text = text[2:]
    return text.replace("_", "-")


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_tx_ids(text: str) -> list[str]:
    return [canonical_tx_id(x) for x in parse_csv(text)]


def _as_str(data: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.asarray([""] * n, dtype=str)
    arr = np.asarray(data[key])
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype=str)
    if arr.shape[0] != n:
        raise ValueError(f"{key} length mismatch: {arr.shape[0]} != {n}")
    return arr.astype(str)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        n = int(features.shape[0])
        logits = np.asarray(data["tx_logits"], dtype=np.float32) if "tx_logits" in data.files else None
        return {
            "path": str(path),
            "features": features,
            "tx_logits": logits,
            "dataset_role": _as_str(data, "dataset_role", n),
            "tx_ids": np.asarray([canonical_tx_id(x) for x in _as_str(data, "tx_ids", n)], dtype=str),
            "rx_ids": _as_str(data, "rx_ids", n),
            "day_ids": _as_str(data, "day_ids", n),
            "eq_ids": _as_str(data, "eq_ids", n),
            "sig_ids": _as_str(data, "sig_ids", n),
            "sat_scenarios": _as_str(data, "sat_scenarios", n),
            "channel_views": _as_str(data, "channel_views", n),
        }


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-6)


def make_prototypes(clean_npz: Path, source_tx_ids: list[str]) -> np.ndarray:
    payload = load_npz(clean_npz)
    feats = payload["features"]
    tx = payload["tx_ids"]
    protos = []
    for item in source_tx_ids:
        mask = tx == canonical_tx_id(item)
        if not mask.any():
            raise ValueError(f"no source clean prototype rows for tx={item}")
        proto = feats[mask].mean(axis=0)
        proto = proto / max(float(np.linalg.norm(proto)), 1.0e-6)
        protos.append(proto.astype(np.float32))
    return np.stack(protos, axis=0)


def top_margin(logits: np.ndarray) -> np.ndarray:
    if logits.shape[1] < 2:
        return np.zeros(logits.shape[0], dtype=np.float64)
    part = np.partition(logits.astype(np.float64), -2, axis=1)
    return part[:, -1] - part[:, -2]


def safe_rate(num: int, den: int) -> float:
    return float("nan") if den <= 0 else float(num) / float(den)


def safe_mean(values: np.ndarray) -> float:
    return float("nan") if values.size == 0 else float(np.mean(values))


def safe_min(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float("nan") if not vals else float(min(vals))


def subgroup_acc(pred_tx: np.ndarray, true_tx: np.ndarray, mask: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    out = {}
    for group in sorted(set(groups[mask].astype(str))):
        if group == "":
            continue
        gmask = mask & (groups.astype(str) == group)
        out[group] = safe_rate(int((pred_tx[gmask] == true_tx[gmask]).sum()), int(gmask.sum()))
    return out


def source_accept_far(oldness: np.ndarray, source_mask: np.ndarray, unknown_mask: np.ndarray, q: float) -> float:
    if not source_mask.any() or not unknown_mask.any():
        return float("nan")
    threshold = float(np.quantile(oldness[source_mask], float(q)))
    return safe_rate(int((oldness[unknown_mask] >= threshold).sum()), int(unknown_mask.sum()))


def evaluate_payload(payload: dict, prototypes: np.ndarray, source_tx_ids: list[str], *, clean_reference_acc: float, identity_ref: dict | None, args: argparse.Namespace) -> dict:
    tx_to_idx = {tx: i for i, tx in enumerate(source_tx_ids)}
    source_old_tx = set(source_tx_ids)
    features = l2n(payload["features"])
    proto_logits = features @ prototypes.T
    logits = payload["tx_logits"] if payload["tx_logits"] is not None else proto_logits
    pred_idx = np.asarray(logits).argmax(axis=1)
    pred_tx = np.asarray([source_tx_ids[int(i)] for i in pred_idx], dtype=str)
    true_tx = payload["tx_ids"].astype(str)
    roles = payload["dataset_role"].astype(str)
    known = (roles == "target_old") & np.asarray([x in source_old_tx for x in true_tx], dtype=bool)
    source = (roles == "source") & np.asarray([x in source_old_tx for x in true_tx], dtype=bool)
    proxy_unknown = roles == "proxy_unknown"
    target_unknown = roles == "target_unknown"
    unknown_any = proxy_unknown | target_unknown

    source_acc = safe_rate(int((pred_tx[source] == true_tx[source]).sum()), int(source.sum()))
    target_acc = safe_rate(int((pred_tx[known] == true_tx[known]).sum()), int(known.sum()))
    scenario_acc = subgroup_acc(pred_tx, true_tx, known, payload["sat_scenarios"])
    rx_acc = subgroup_acc(pred_tx, true_tx, known, payload["rx_ids"])
    tx_acc = subgroup_acc(pred_tx, true_tx, known, true_tx)

    max_logit = np.asarray(logits, dtype=np.float64).max(axis=1)
    margin = top_margin(np.asarray(logits, dtype=np.float64))
    proto_pred = proto_logits.argmax(axis=1)
    proto_pred_tx = np.asarray([source_tx_ids[int(i)] for i in proto_pred], dtype=str)
    proto_target_acc = safe_rate(int((proto_pred_tx[known] == true_tx[known]).sum()), int(known.sum()))

    true_idx = np.asarray([tx_to_idx.get(tx, -1) for tx in true_tx], dtype=np.int64)
    valid_true = true_idx >= 0
    true_sim = np.full(features.shape[0], np.nan, dtype=np.float64)
    true_cos_dist = np.full(features.shape[0], np.nan, dtype=np.float64)
    proto_margin = np.full(features.shape[0], np.nan, dtype=np.float64)
    for i in np.where(valid_true)[0]:
        cls = int(true_idx[i])
        sims = proto_logits[i].astype(np.float64)
        true_sim[i] = float(sims[cls])
        true_cos_dist[i] = float(1.0 - sims[cls])
        others = np.delete(sims, cls)
        proto_margin[i] = float(sims[cls] - others.max()) if others.size else float("nan")

    far_source05_target = source_accept_far(max_logit, source, target_unknown, 0.05)
    far_source05_proxy = source_accept_far(max_logit, source, proxy_unknown, 0.05)
    out = {
        "source_acc": source_acc,
        "target_old_closed_acc": target_acc,
        "target_old_proto_acc": proto_target_acc,
        "target_old_delta_pp_vs_clean": 100.0 * (target_acc - clean_reference_acc) if np.isfinite(clean_reference_acc) else float("nan"),
        "target_old_mean_margin": safe_mean(margin[known]),
        "target_old_mean_proto_margin": safe_mean(proto_margin[known]),
        "target_old_mean_true_cos_dist": safe_mean(true_cos_dist[known]),
        "target_old_min_scenario_acc": safe_min(scenario_acc.values()),
        "target_old_min_rx_acc": safe_min(rx_acc.values()),
        "target_old_min_tx_acc": safe_min(tx_acc.values()),
        "target_unknown_mean_oldness": safe_mean(max_logit[target_unknown]),
        "proxy_unknown_mean_oldness": safe_mean(max_logit[proxy_unknown]),
        "target_unknown_far_source05": far_source05_target,
        "proxy_unknown_far_source05": far_source05_proxy,
        "known_count": int(known.sum()),
        "source_count": int(source.sum()),
        "proxy_unknown_count": int(proxy_unknown.sum()),
        "target_unknown_count": int(target_unknown.sum()),
        "scenario_acc_json": json.dumps(scenario_acc, ensure_ascii=True, sort_keys=True),
        "rx_acc_json": json.dumps(rx_acc, ensure_ascii=True, sort_keys=True),
        "tx_acc_json": json.dumps(tx_acc, ensure_ascii=True, sort_keys=True),
    }
    if identity_ref:
        out.update({
            "target_old_delta_pp_vs_identity": 100.0 * (target_acc - float(identity_ref["target_old_closed_acc"])),
            "min_scenario_delta_pp_vs_identity": 100.0 * (out["target_old_min_scenario_acc"] - float(identity_ref["target_old_min_scenario_acc"])),
            "min_tx_delta_pp_vs_identity": 100.0 * (out["target_old_min_tx_acc"] - float(identity_ref["target_old_min_tx_acc"])),
            "target_unknown_oldness_delta_vs_identity": out["target_unknown_mean_oldness"] - float(identity_ref["target_unknown_mean_oldness"]),
            "proxy_unknown_oldness_delta_vs_identity": out["proxy_unknown_mean_oldness"] - float(identity_ref["proxy_unknown_mean_oldness"]),
            "target_unknown_far_delta_vs_identity": out["target_unknown_far_source05"] - float(identity_ref["target_unknown_far_source05"]),
            "target_old_margin_delta_vs_identity": out["target_old_mean_margin"] - float(identity_ref["target_old_mean_margin"]),
            "target_old_true_dist_delta_vs_identity": out["target_old_mean_true_cos_dist"] - float(identity_ref["target_old_mean_true_cos_dist"]),
        })
    else:
        out.update({
            "target_old_delta_pp_vs_identity": float("nan"),
            "min_scenario_delta_pp_vs_identity": float("nan"),
            "min_tx_delta_pp_vs_identity": float("nan"),
            "target_unknown_oldness_delta_vs_identity": float("nan"),
            "proxy_unknown_oldness_delta_vs_identity": float("nan"),
            "target_unknown_far_delta_vs_identity": float("nan"),
            "target_old_margin_delta_vs_identity": float("nan"),
            "target_old_true_dist_delta_vs_identity": float("nan"),
        })

    old_recovery_pass = bool(
        out["target_old_closed_acc"] >= float(args.target_acc_floor)
        or out["target_old_delta_pp_vs_identity"] >= float(args.min_delta_pp)
    )
    floor_pass = bool(
        out["target_old_min_scenario_acc"] >= float(args.scenario_floor)
        and out["target_old_min_tx_acc"] >= float(args.tx_floor)
    )
    margin_pass = bool(
        not identity_ref
        or (
            out["target_old_margin_delta_vs_identity"] >= -float(args.margin_tolerance)
            and out["target_old_true_dist_delta_vs_identity"] <= float(args.distance_tolerance)
        )
    )
    unknown_pass = bool(
        not identity_ref
        or (
            out["target_unknown_far_delta_vs_identity"] <= float(args.far_tolerance)
            and out["target_unknown_oldness_delta_vs_identity"] <= float(args.oldness_tolerance)
        )
    )
    out.update({
        "passes_old_recovery_gate": old_recovery_pass,
        "passes_floor_gate": floor_pass,
        "passes_margin_gate": margin_pass,
        "passes_unknown_safety_gate": unknown_pass,
        "passes_strong_target1": bool(old_recovery_pass and floor_pass and margin_pass and unknown_pass),
    })
    return out


def clean_reference(run_dir: Path, source_tx_ids: list[str], clean_relpath: str) -> dict:
    payload = load_npz(run_dir / clean_relpath)
    logits = payload["tx_logits"]
    if logits is None:
        raise ValueError(f"{run_dir / clean_relpath} lacks tx_logits")
    pred = np.asarray([source_tx_ids[int(i)] for i in logits.argmax(axis=1)], dtype=str)
    roles = payload["dataset_role"].astype(str)
    tx = payload["tx_ids"].astype(str)
    known = roles == "target_old"
    return {
        "clean_target_old_acc": safe_rate(int((pred[known] == tx[known]).sum()), int(known.sum())),
        "clean_target_old_count": int(known.sum()),
        "clean_target_old_min_tx_acc": safe_min(subgroup_acc(pred, tx, known, tx).values()),
    }


def clean_repaired_metrics(path: Path, source_tx_ids: list[str], clean_ref_acc: float) -> dict:
    if not path.is_file():
        return {
            "clean_adapter_eval_available": False,
            "clean_repaired_target_old_acc": float("nan"),
            "clean_repaired_drop_pp": float("nan"),
            "passes_clean_fidelity_gate": False,
        }
    payload = load_npz(path)
    logits = payload["tx_logits"]
    if logits is None:
        return {
            "clean_adapter_eval_available": False,
            "clean_repaired_target_old_acc": float("nan"),
            "clean_repaired_drop_pp": float("nan"),
            "passes_clean_fidelity_gate": False,
        }
    pred = np.asarray([source_tx_ids[int(i)] for i in logits.argmax(axis=1)], dtype=str)
    roles = payload["dataset_role"].astype(str)
    tx = payload["tx_ids"].astype(str)
    known = roles == "target_old"
    acc = safe_rate(int((pred[known] == tx[known]).sum()), int(known.sum()))
    drop = 100.0 * (float(clean_ref_acc) - acc) if np.isfinite(float(clean_ref_acc)) and np.isfinite(acc) else float("nan")
    return {
        "clean_adapter_eval_available": True,
        "clean_repaired_target_old_acc": acc,
        "clean_repaired_drop_pp": drop,
        "passes_clean_fidelity_gate": bool(drop <= 2.0),
    }


def evaluate_run(run_dir: Path, prototypes: np.ndarray, source_tx_ids: list[str], args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    clean_ref = clean_reference(run_dir, source_tx_ids, str(args.clean_relpath))
    identity_payload_path = run_dir / str(args.identity_relpath)
    if not identity_payload_path.is_file():
        return rows
    identity_payload = load_npz(identity_payload_path)
    identity_metrics = evaluate_payload(identity_payload, prototypes, source_tx_ids, clean_reference_acc=clean_ref["clean_target_old_acc"], identity_ref=None, args=args)
    identity_metrics.update({
        "run_id": run_dir.name,
        "variant": "LEOADAPT3_IDENTITY",
        "feature_path": str(identity_payload_path),
        **clean_ref,
        "clean_adapter_eval_available": False,
        "clean_repaired_target_old_acc": float("nan"),
        "clean_repaired_drop_pp": float("nan"),
        "passes_clean_fidelity_gate": False,
        "verdict": "identity_baseline",
    })
    rows.append(identity_metrics)

    variants = parse_csv(args.variants)
    identity_ref = identity_metrics
    for variant in variants:
        rel = str(args.feature_relpath).format(variant=variant)
        path = run_dir / rel
        if not path.is_file():
            continue
        payload = load_npz(path)
        metrics = evaluate_payload(payload, prototypes, source_tx_ids, clean_reference_acc=clean_ref["clean_target_old_acc"], identity_ref=identity_ref, args=args)
        clean_rel = str(args.clean_repaired_relpath).format(variant=variant)
        clean_metrics = clean_repaired_metrics(run_dir / clean_rel, source_tx_ids, clean_ref["clean_target_old_acc"])
        metrics.update(clean_metrics)
        metrics["passes_strong_target1"] = bool(metrics["passes_strong_target1"] and clean_metrics["passes_clean_fidelity_gate"])
        verdict = "pass" if metrics["passes_strong_target1"] else "fail"
        metrics.update({
            "run_id": run_dir.name,
            "variant": variant,
            "feature_path": str(path),
            **clean_ref,
            "verdict": verdict,
        })
        rows.append(metrics)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["run_id"]
    leading = ["run_id", "variant", "verdict", "passes_strong_target1", "target_old_closed_acc", "target_old_delta_pp_vs_identity", "clean_repaired_drop_pp", "target_old_min_scenario_acc", "target_old_min_tx_acc", "target_unknown_far_source05"]
    fields = leading + [f for f in fields if f not in leading]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--source_clean_npz", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--metrics_json", type=Path, required=True)
    p.add_argument("--run_glob", default="phase1_adv3b02_multiview_keepold_*_20260702")
    p.add_argument("--source_tx_ids", default="14-10,14-7,20-15,20-19,6-15,8-20")
    p.add_argument("--clean_relpath", default="ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz")
    p.add_argument("--identity_relpath", default="LEOADAPT3_IDENTITY/features_leo_repaired.npz")
    p.add_argument("--feature_relpath", default="{variant}/features_leo_repaired.npz")
    p.add_argument("--clean_repaired_relpath", default="{variant}/features_clean_repaired.npz")
    p.add_argument("--variants", default="LEOADAPT3_IDENTITY,LEOADAPT3_MEANSHIFT,LEOADAPT3_NORMSHIFT,LEOADAPT3_LINR_COS,LEOADAPT3_MLP_ID")
    p.add_argument("--target_acc_floor", type=float, default=0.80)
    p.add_argument("--min_delta_pp", type=float, default=5.0)
    p.add_argument("--scenario_floor", type=float, default=0.80)
    p.add_argument("--tx_floor", type=float, default=0.70)
    p.add_argument("--margin_tolerance", type=float, default=0.02)
    p.add_argument("--distance_tolerance", type=float, default=0.02)
    p.add_argument("--far_tolerance", type=float, default=0.0)
    p.add_argument("--oldness_tolerance", type=float, default=0.0)
    args = p.parse_args()

    source_tx_ids = parse_tx_ids(args.source_tx_ids)
    prototypes = make_prototypes(args.source_clean_npz, source_tx_ids)
    rows = []
    for run_dir in sorted(args.runs_root.glob(str(args.run_glob))):
        rows.extend(evaluate_run(run_dir, prototypes, source_tx_ids, args))
    write_csv(args.out_csv, rows)
    candidates = [r for r in rows if r.get("variant") != "LEOADAPT3_IDENTITY"]
    pass_rows = [r for r in candidates if bool(r.get("passes_strong_target1"))]
    best_by_acc = sorted(candidates, key=lambda r: (-float(r["target_old_closed_acc"]), float(r["target_unknown_far_source05"])))[:10]
    best_by_delta = sorted(candidates, key=lambda r: (-float(r["target_old_delta_pp_vs_identity"]), float(r["target_unknown_far_delta_vs_identity"])))[:10]
    best_unknown_safe = sorted(candidates, key=lambda r: (float(r["target_unknown_far_delta_vs_identity"]), -float(r["target_old_delta_pp_vs_identity"])))[:10]
    summary = {
        "phase": "phase1_target1_strong_repair_audit_v17",
        "rows": len(rows),
        "candidate_rows": len(candidates),
        "strong_target1_pass": len(pass_rows),
        "out_csv": str(args.out_csv),
        "uses_target_clean_for_training": False,
        "uses_target_labels_for_training": False,
        "target_labels_evaluation_only": True,
        "clean_adapter_eval_available": False,
        "thresholds": {
            "target_acc_floor": float(args.target_acc_floor),
            "min_delta_pp": float(args.min_delta_pp),
            "scenario_floor": float(args.scenario_floor),
            "tx_floor": float(args.tx_floor),
            "unknown_far_delta_max": float(args.far_tolerance),
            "unknown_oldness_delta_max": float(args.oldness_tolerance),
        },
        "pass_rows": pass_rows[:20],
        "best_by_target_old_acc": best_by_acc,
        "best_by_delta": best_by_delta,
        "best_unknown_safe": best_unknown_safe,
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["rows", "candidate_rows", "strong_target1_pass"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
