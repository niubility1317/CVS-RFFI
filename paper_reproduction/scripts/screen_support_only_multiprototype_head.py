#!/usr/bin/env python3
"""Diagnostic-only SOMP-H screen on target-only LEO feature caches.

The diagnostic NPZ contains dataset roles and TX truth, so this script is never
a formal Phase2 predictor.  It hard-rejects any cache that co-resides with a
clean view before loading its feature tensor.  Role/truth fields are used only
to construct the declared split and to score immutable predictions afterwards.
The fitted head receives registered support features and labels only; query
features are scored independently over all registered classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from paper_reproduction.cvs_aligned.support_only_multiprototype_head import (
    fit_support_only_multiprototype_head,
    pack_support_only_multiprototype_head,
    predict_support_only_multiprototype_head,
    unpack_support_only_multiprototype_head,
)


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ADV3B02_SHA256 = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stable_rank(seed: int, *parts: object) -> int:
    raw = ":".join([str(seed), *(str(value) for value in parts)])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _sha256_values(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_id(arrays: dict[str, np.ndarray], index: int) -> str:
    return "|".join(
        str(arrays[key][index])
        for key in ("dataset_role", "tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids")
    )


def _load_cache(path: Path, *, scenario: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "features",
            "tx_ids",
            "rx_ids",
            "day_ids",
            "eq_ids",
            "sig_ids",
            "dataset_role",
            "channel_views",
            "sat_scenarios",
            "manifest_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing members {missing}")
        # Inspect only small provenance arrays first.  A clean row anywhere in
        # the package makes the whole path ineligible for Phase2; do not load
        # the feature tensor and then try to mask the row away.
        channel_views = np.asarray(archive["channel_views"])
        sat_scenarios = np.asarray(archive["sat_scenarios"])
        if any("clean" in value.lower() for value in channel_views.astype(str).tolist()):
            raise ValueError(
                f"{path}: PROTOCOL_INVALID_FOR_PHASE2 because clean view co-resides in cache"
            )
        nonempty_scenarios = {
            value for value in sat_scenarios.astype(str).tolist() if value
        }
        if nonempty_scenarios != {scenario}:
            raise ValueError(
                f"{path}: nonempty scenario rows are not exclusively {scenario}"
            )
        arrays = {key: np.asarray(archive[key]) for key in required - {"manifest_json"}}
        manifest = json.loads(str(archive["manifest_json"].item()))
    row_count = len(arrays["features"])
    if any(len(value) != row_count for key, value in arrays.items() if key != "features"):
        raise ValueError(f"{path}: cache arrays have inconsistent row counts")
    if arrays["features"].ndim != 2 or not np.isfinite(arrays["features"]).all():
        raise ValueError(f"{path}: invalid feature matrix")
    if manifest.get("source_checkpoint_sha256") != ADV3B02_SHA256:
        raise ValueError(f"{path}: cache is not from the locked ADV3B02 checkpoint")
    declared = set(manifest.get("target_channel_scenarios", []))
    if scenario not in declared:
        raise ValueError(f"{path}: manifest does not declare {scenario}")
    return arrays, manifest


def _select_split(
    arrays: dict[str, np.ndarray],
    *,
    role: str,
    tx_labels: list[str],
    receiver: str,
    seed: int,
    k_shot: int,
    support_pool_max_k: int,
    query_per_tx: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    roles = arrays["dataset_role"].astype(str)
    tx = arrays["tx_ids"].astype(str)
    rx = arrays["rx_ids"].astype(str)
    support: list[int] = []
    query: list[int] = []
    per_tx: dict[str, Any] = {}
    for label in tx_labels:
        candidates = np.where((roles == role) & (tx == label) & (rx == receiver))[0]
        ordered = sorted(
            (int(index) for index in candidates.tolist()),
            key=lambda index: _stable_rank(
                seed,
                role,
                label,
                receiver,
                arrays["day_ids"][index],
                arrays["eq_ids"][index],
                arrays["sig_ids"][index],
            ),
        )
        needed = int(support_pool_max_k) + int(query_per_tx)
        if len(ordered) < needed:
            raise ValueError(
                f"insufficient {role}/{label}/{receiver}: {len(ordered)} < {needed}"
            )
        class_support = ordered[: int(k_shot)]
        class_query = ordered[int(support_pool_max_k) : needed]
        support.extend(class_support)
        query.extend(class_query)
        per_tx[label] = {
            "available": len(ordered),
            "support_ids": [_sample_id(arrays, index) for index in class_support],
            "query_ids": [_sample_id(arrays, index) for index in class_query],
        }
    support_array = np.asarray(support, dtype=np.int64)
    query_array = np.asarray(query, dtype=np.int64)
    if set(support_array.tolist()) & set(query_array.tolist()):
        raise AssertionError("support/query overlap")
    support_ids = [_sample_id(arrays, int(index)) for index in support_array]
    query_ids = [_sample_id(arrays, int(index)) for index in query_array]
    return support_array, query_array, {
        "role": role,
        "receiver": receiver,
        "seed": int(seed),
        "k_shot": int(k_shot),
        "support_pool_max_k": int(support_pool_max_k),
        "query_per_tx": int(query_per_tx),
        "support_ids_sha256": _sha256_values(support_ids),
        "query_ids_sha256": _sha256_values(query_ids),
        "support_query_disjoint": True,
        "per_tx": per_tx,
    }


def _encode(labels: np.ndarray, class_order: list[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(class_order)}
    try:
        return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"label outside registered class order: {exc.args[0]}") from exc


def _accuracy_by_class(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    for label in sorted(set(truth.astype(str).tolist())):
        mask = truth.astype(str) == label
        output[label] = float(np.mean(predicted[mask].astype(str) == label))
    return output


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right <= 0.0 else float(2.0 * left * right / (left + right))


def _fit_predict(
    support_x: np.ndarray,
    support_y: np.ndarray,
    query_x: np.ndarray,
    *,
    class_order: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    encoded = _encode(support_y, class_order)
    started = time.perf_counter()
    head = fit_support_only_multiprototype_head(
        support_x,
        encoded,
        class_count=len(class_order),
    )
    fit_seconds = time.perf_counter() - started
    packed = pack_support_only_multiprototype_head(head)
    head = unpack_support_only_multiprototype_head(packed)
    started = time.perf_counter()
    prediction_ids = predict_support_only_multiprototype_head(query_x, head)
    predict_seconds = time.perf_counter() - started
    return np.asarray(class_order, dtype=str)[prediction_ids], {
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "query_count": int(len(query_x)),
        "seconds_per_query": float(predict_seconds / max(1, len(query_x))),
        "trainable_parameters": 0,
        "adaptation_epochs": 0,
        "persistent_state_bytes_fp16": int(head.persistent_state_bytes_fp16),
        "extra_macs_per_query": int(head.extra_macs_per_query),
        "prototype_count": int(head.prototype_count),
        "feature_dim": int(head.feature_dim),
        "packed_numeric_tensor_bytes": int(
            sum(
                value.nbytes
                for key, value in packed.items()
                if key not in {"schema_utf8", "support_audit_json_utf8"}
            )
        ),
        "support_audit": head.support_audit,
    }


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_json(Path(args.config))
    if config.get("phase2_sample_view_policy") != "leo_weak_only_no_clean_access":
        raise ValueError("row config does not declare leo_weak_only_no_clean_access")
    if config.get("clean_sample_access") is not False:
        raise ValueError("row config does not deny clean sample access")
    if config.get("clean_derived_signal_access") is not False:
        raise ValueError("row config does not deny clean-derived signals")
    old_labels = [str(value) for value in config["target_old_tx_labels"]]
    locked_new = [str(value) for value in config["target_new_tx_labels"]]
    requested_counts = [int(value) for value in args.new_counts.split(",")]
    if any(value not in {5, 10, 20} for value in requested_counts):
        raise ValueError("new_counts must be a comma-separated subset of 5,10,20")
    if max(requested_counts) > len(locked_new):
        raise ValueError("row config does not cover the requested nested new classes")
    if int(args.k_shot) not in {5, 10}:
        raise ValueError("formal working points are K=5 or K=10")
    feature_root = Path(args.feature_root)
    rows: list[dict[str, Any]] = []
    split_evidence: dict[str, Any] = {}
    cache_evidence: dict[str, Any] = {}
    for scenario in SCENARIOS:
        cache_path = feature_root / f"{scenario}.npz"
        arrays, manifest = _load_cache(cache_path, scenario=scenario)
        cache_evidence[scenario] = {
            "path": str(cache_path.resolve()),
            "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
            "channel_views": sorted(set(arrays["channel_views"].astype(str).tolist())),
            "sat_scenarios": sorted(set(arrays["sat_scenarios"].astype(str).tolist())),
            "formal_predictor_eligible": False,
            "ineligibility_reasons": [
                "query_truth_and_dataset_role_co_resident_in_legacy_npz",
                "raw_dataset_paths_present_in_embedded_build_manifest",
                "no_current_os_access_ledger_bound_to_this_diagnostic_run",
            ],
        }
        old_support, old_query, old_split = _select_split(
            arrays,
            role="target_old",
            tx_labels=old_labels,
            receiver=str(args.receiver),
            seed=int(args.seed),
            k_shot=int(args.k_shot),
            support_pool_max_k=int(config["support_pool_max_k"]),
            query_per_tx=int(config["query_per_tx"]),
        )
        split_evidence[f"{scenario}:old"] = old_split
        old_truth = arrays["tx_ids"][old_query].astype(str)
        before_prediction, before_resource = _fit_predict(
            arrays["features"][old_support],
            arrays["tx_ids"][old_support],
            arrays["features"][old_query],
            class_order=old_labels,
        )
        before_by_class = _accuracy_by_class(old_truth, before_prediction)
        before_old = float(np.mean(before_prediction == old_truth))
        for new_count in requested_counts:
            new_labels = locked_new[:new_count]
            new_support, new_query, new_split = _select_split(
                arrays,
                role="target_new",
                tx_labels=new_labels,
                receiver=str(args.receiver),
                seed=int(args.seed),
                k_shot=int(args.k_shot),
                support_pool_max_k=int(config["support_pool_max_k"]),
                query_per_tx=int(config["query_per_tx"]),
            )
            split_evidence[f"{scenario}:new{new_count}"] = new_split
            support_indices = np.concatenate([old_support, new_support])
            query_indices = np.concatenate([old_query, new_query])
            class_order = [*old_labels, *new_labels]
            prediction, resource = _fit_predict(
                arrays["features"][support_indices],
                arrays["tx_ids"][support_indices],
                arrays["features"][query_indices],
                class_order=class_order,
            )
            truth = arrays["tx_ids"][query_indices].astype(str)
            old_mask = np.isin(truth, old_labels)
            new_mask = np.isin(truth, new_labels)
            after_old = float(np.mean(prediction[old_mask] == truth[old_mask]))
            seen_new = float(np.mean(prediction[new_mask] == truth[new_mask]))
            per_class = _accuracy_by_class(truth, prediction)
            rows.append(
                {
                    "scenario": scenario,
                    "receiver": str(args.receiver),
                    "seed": int(args.seed),
                    "k_shot": int(args.k_shot),
                    "new_class_count": int(new_count),
                    "before_target_old_acc": before_old,
                    "before_min_old_class_acc": min(before_by_class.values()),
                    "after_target_old_acc": after_old,
                    "after_min_old_class_acc": min(per_class[label] for label in old_labels),
                    "seen_new_acc": seen_new,
                    "h_old_new": _harmonic(after_old, seen_new),
                    "old_forgetting": before_old - after_old,
                    "before_old_per_class": before_by_class,
                    "after_all_per_class": per_class,
                    "before_resource": before_resource,
                    "after_resource": resource,
                }
            )
    aggregate: list[dict[str, Any]] = []
    for new_count in requested_counts:
        selected = [row for row in rows if row["new_class_count"] == new_count]
        keys = (
            "before_target_old_acc",
            "before_min_old_class_acc",
            "after_target_old_acc",
            "after_min_old_class_acc",
            "seen_new_acc",
            "h_old_new",
            "old_forgetting",
        )
        aggregate.append(
            {
                "new_class_count": int(new_count),
                **{key: float(np.mean([row[key] for row in selected])) for key in keys},
            }
        )
    return {
        "schema": "cvs.stage2bc.somph_diagnostic_screen.v1",
        "status": "DIAGNOSTIC_ONLY_NOT_FORMAL_PHASE2_EVIDENCE",
        "method": "SOMP-H",
        "base_model": "ADV3B02_CORE90_SOFT_E200",
        "base_checkpoint_sha256": ADV3B02_SHA256,
        "protocol_boundary": {
            "support_and_query_channel_view": "leo_weak_only",
            "clean_samples_used_by_head": False,
            "clean_derived_signals_used_by_head": False,
            "query_labels_used_by_head_fit": False,
            "query_roles_used_by_head_fit_or_predict": False,
            "query_class_quota_used": False,
            "query_global_assignment_used": False,
            "dense_query_graph_used": False,
            "formal_evidence_allowed": False,
        },
        "config": str(Path(args.config).resolve()),
        "rows": rows,
        "aggregate_across_three_leo_weak_scenarios": aggregate,
        "split_evidence": split_evidence,
        "cache_evidence": cache_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receiver", default="20-1")
    parser.add_argument("--seed", type=int, default=713101)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--new-counts", default="5,10,20")
    args = parser.parse_args()
    output = run_screen(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["aggregate_across_three_leo_weak_scenarios"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
