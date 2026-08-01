#!/usr/bin/env python3
"""Run the minimal D110-SCPM G0 check on one pinned real D106 tap.

The entry reuses the established D106 one-shot archive, fold, baseline, bitmap,
and immutable-output helpers.  It adds only the frozen D106 closed rank-three
basis, the D110 four-scalar prior, and support-only SCPM scoring.  No query
label or performance value is consumed or emitted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import run_d106_rcmr_g0_one_shot as base  # noqa: E402
from cvsrffi import stage2_d106_rdce_asset as d106_asset  # noqa: E402
from cvsrffi import stage2_d110_scpm_asset as scpm_asset  # noqa: E402
from cvsrffi import stage2_d110_scpm_runtime as scpm_runtime  # noqa: E402


ONE_SHOT_SCHEMA = "cvs.phase1.d110.scpm_g0.one_shot.v1"
ONE_SHOT_STATUS = "REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE"
CANDIDATE_ID = scpm_asset.CANDIDATE_ID


class D110SCPMG0Error(ValueError):
    """Raised when the thin SCPM G0 execution drifts."""


def _geometry(rows: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reproduce the deployed quantized D106 basis and D110 12-byte prior."""

    count = base.g0.EXPECTED_ROWS
    tx = d106_asset._typed_tokens(rows.tx_labels, "tap.tx_labels", count)
    receiver = d106_asset._typed_tokens(
        rows.receiver_ids, "tap.receiver_ids", count
    )
    day = d106_asset._typed_tokens(rows.day_ids, "tap.day_ids", count)
    raw_basis, _scatter, _groups, class_count = d106_asset._build_geometry(
        rows.z_id, tx, receiver, day
    )
    if class_count != base.g0.EXPECTED_CLASSES:
        raise D110SCPMG0Error("D110 G0 source class count drift")
    basis_codes, basis_scales = d106_asset._quantize_basis(raw_basis)
    closed_u = d106_asset._orthogonal_closure(basis_codes, basis_scales)
    raw_prior = scpm_asset._cell_conditional_variances(
        rows.z_id, tx, receiver, day, closed_u
    )
    prior_codes, prior_scales, relative_error = (
        scpm_asset._quantize_positive_variances(raw_prior)
    )
    prior = scpm_asset._decode_positive_variances(prior_codes, prior_scales)
    receipt = {
        "closed_basis_root_sha256": base.g0._array_root(closed_u),
        "basis_codes_root_sha256": base.g0._array_root(basis_codes),
        "basis_scales_root_sha256": base.g0._array_root(basis_scales),
        "prior_codes_root_sha256": base.g0._array_root(prior_codes),
        "prior_scales_root_sha256": base.g0._array_root(prior_scales),
        "prior_root_sha256": base.g0._array_root(prior),
        "prior_quantized_bytes": int(prior_codes.nbytes + prior_scales.nbytes),
        "prior_quantization_max_relative_error": float(relative_error),
    }
    receipt["geometry_root_sha256"] = base.g0._sha256(receipt)
    return closed_u, prior, receipt


def _metric_features(
    rows: np.ndarray, closed_u: np.ndarray, predictive: np.ndarray
) -> np.ndarray:
    normalized = scpm_runtime._l2_normalize_rows(rows, "D110 G0 features")
    projected = normalized @ closed_u.T
    parallel = projected @ closed_u
    perpendicular = normalized - parallel
    transformed = (
        (projected / np.sqrt(predictive[:-1])[None, :]) @ closed_u
        + perpendicular / float(np.sqrt(predictive[-1]))
    )
    if not np.isfinite(transformed).all():
        raise D110SCPMG0Error("D110 G0 metric feature became non-finite")
    return np.ascontiguousarray(transformed, dtype=np.float64)


def _candidate_trace(
    inputs: Mapping[str, Any],
    *,
    registry: tuple[str, ...],
    closed_u: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, dict[str, Any]]:
    support = np.asarray(inputs["support_plus"], dtype=np.float32)
    labels = np.asarray(tuple(inputs["support_labels"]), dtype="<U64")
    queries = np.asarray(inputs["query_plus"], dtype=np.float32)
    state = scpm_runtime.fit_d110_scpm_runtime(
        support, labels, closed_u, prior
    )
    if tuple(state.class_labels.tolist()) != registry:
        raise D110SCPMG0Error("D110 G0 class order drift")
    logits = -np.stack(
        [scpm_runtime.score_d110_scpm_query(state, query) for query in queries],
        axis=0,
    )
    query_features = _metric_features(queries, closed_u, state.predictive_variances)
    support_features = _metric_features(
        support, closed_u, state.predictive_variances
    )
    # Avoid a [query,support,d] temporary: the bounded pairwise matrix is enough.
    squared_distance = (
        np.sum(np.square(query_features), axis=1)[:, None]
        + np.sum(np.square(support_features), axis=1)[None, :]
        - 2.0 * (query_features @ support_features.T)
    )
    support_contributions = -np.maximum(squared_distance, 0.0)
    numeric_bytes = sum(
        int(value.nbytes)
        for value in (
            state.class_labels,
            state.centers,
            state.closed_u,
            state.prior_variances,
            state.variances,
            state.safe_relative_variances,
            state.predictive_variances,
        )
    ) + (0 if state.target_variances is None else int(state.target_variances.nbytes))
    summary = {
        "active_k": state.active_k,
        "alpha_binary64": float(state.alpha).hex(),
        "euclidean_fallback": state.euclidean_fallback,
        "runtime_state_numeric_bytes": numeric_bytes,
        "query_rows_used_for_fit": state.query_rows_used_for_fit,
        "query_state_updates": state.query_state_updates,
    }
    summary["runtime_state_root_sha256"] = base.g0._sha256(summary)
    return (
        np.ascontiguousarray(logits, dtype=np.float64),
        np.ascontiguousarray(support_contributions, dtype=np.float64),
        tuple(inputs["support_ids"]),
        query_features,
        summary,
    )


def _fold_audit(
    snapshot: Any,
    fold: Any,
    *,
    active_k: int,
    predecessor_lock: Any,
    registry: tuple[str, ...],
    closed_u: np.ndarray,
    prior: np.ndarray,
) -> dict[str, Any]:
    inputs = base._fold_inputs(snapshot, fold, active_k=active_k, registry=registry)
    baseline_logits, baseline_kernels, baseline_ids, baseline_features = (
        base._baseline_trace(
            inputs, registry=registry, predecessor_lock=predecessor_lock
        )
    )
    candidate_logits, candidate_terms, candidate_ids, candidate_features, state = (
        _candidate_trace(
            inputs, registry=registry, closed_u=closed_u, prior=prior
        )
    )
    baseline_argmax = base.g0._unique_argmax(baseline_logits, registry)
    candidate_argmax = base._argmax_labels(candidate_logits, registry)
    baseline_neighbors = base._dominant_support_signatures(
        baseline_kernels, baseline_ids
    )
    candidate_neighbors = base._dominant_support_signatures(
        candidate_terms, candidate_ids
    )
    baseline_margin = base._top_two_margins(baseline_logits.astype(np.float64))
    candidate_margin = base._top_two_margins(candidate_logits)
    query_root = fold.query_root_sha256
    feature = base._array_changed_metric(
        metric="scpm_metric_feature",
        query_root_sha256=query_root,
        baseline=baseline_features,
        candidate=candidate_features,
    )
    neighbor = base._changed_metric(
        metric="nearest_support",
        query_root_sha256=query_root,
        baseline=baseline_neighbors,
        candidate=candidate_neighbors,
    )
    margin = base._array_changed_metric(
        metric="top1_minus_top2_margin_binary64",
        query_root_sha256=query_root,
        baseline=baseline_margin.view(np.uint64),
        candidate=candidate_margin.view(np.uint64),
    )
    argmax = base._changed_metric(
        metric="argmax_class",
        query_root_sha256=query_root,
        baseline=baseline_argmax,
        candidate=candidate_argmax,
    )
    payload: dict[str, Any] = {
        "query_root_sha256": query_root,
        "fold_execution_receipt_sha256": base.g0._sha256(
            {
                "fold_id": fold.fold_id,
                "K": active_k,
                "support_root_sha256": base.g0._support_root(inputs["support_ids"]),
                "query_root_sha256": query_root,
                "runtime_state_root_sha256": state["runtime_state_root_sha256"],
            }
        ),
        "runtime_state_numeric_bytes": state["runtime_state_numeric_bytes"],
        "baseline_feature_root_sha256": base.g0._array_root(baseline_features),
        "candidate_feature_root_sha256": base.g0._array_root(candidate_features),
        "baseline_neighbor_root_sha256": base.g0._sha256(
            [list(item) for item in baseline_neighbors]
        ),
        "candidate_neighbor_root_sha256": base.g0._sha256(
            [list(item) for item in candidate_neighbors]
        ),
        "baseline_margin_root_sha256": base.g0._array_root(baseline_margin),
        "candidate_margin_root_sha256": base.g0._array_root(candidate_margin),
        "baseline_argmax_root_sha256": base.g0._sha256(list(baseline_argmax)),
        "candidate_argmax_root_sha256": base.g0._sha256(list(candidate_argmax)),
    }
    for name, value in (
        ("feature", feature),
        ("neighbor", neighbor),
        ("margin", margin),
        ("argmax", argmax),
    ):
        payload[f"{name}_changed_count"] = value["changed_count"]
        payload[f"{name}_changed_bitmap_root_sha256"] = value[
            "changed_bitmap_root_sha256"
        ]
    payload["fold_mechanical_audit_root_sha256"] = base.g0._sha256(payload)
    return payload


def _real_archive_execution(
    rows: Any, *, registered_classes: Sequence[str]
) -> dict[str, Any]:
    registry = base.g0._canonical_registry(
        base._require_registered_classes(registered_classes)
    )
    tap_receipt_sha256 = base._sha256_bytes(base.g0._canonical_bytes(rows.receipt))
    snapshot = base.g0._snapshot_from_rows(
        rows, tap_receipt_sha256=tap_receipt_sha256
    )
    locks = base._validate_one_shot_locks(
        base._predecessor_locks(rows),
        tap_receipt_sha256=snapshot.tap_receipt_sha256,
    )
    closed_u, prior, geometry = _geometry(rows)
    plan = base.g0._build_fold_plan(snapshot)
    query_order = tuple(query for fold in plan for query in fold.query_ids)
    if len(query_order) != base.g0.EXPECTED_ROWS or len(set(query_order)) != len(
        query_order
    ):
        raise D110SCPMG0Error("D110 G0 common query closure drift")
    query_root = base.g0._sha256(list(query_order))
    per_k: list[dict[str, Any]] = []
    for active_k, predecessor_lock in zip(base.g0.K_VALUES, locks, strict=True):
        audits = [
            _fold_audit(
                snapshot,
                fold,
                active_k=active_k,
                predecessor_lock=predecessor_lock,
                registry=registry,
                closed_u=closed_u,
                prior=prior,
            )
            for fold in plan
        ]
        aggregated = base._aggregate_per_k(
            active_k=active_k,
            fold_audits=audits,
            query_root_sha256=query_root,
        )
        aggregated["runtime_state_numeric_bytes_max"] = max(
            int(item["runtime_state_numeric_bytes"]) for item in audits
        )
        per_k.append(aggregated)
    changed = {str(row["K"]): int(row["argmax_changed_count"]) for row in per_k}
    zero_k = [k for k in base.g0.K_VALUES if changed[str(k)] == 0]
    max_support = base.g0.EXPECTED_CLASSES * max(base.g0.K_VALUES)
    max_query = base.g0.MAX_QUERY_ROWS_PER_FOLD
    max_state = max(
        int(row["runtime_state_numeric_bytes_max"]) for row in per_k
    )
    resource_components = {
        "query_metric_features_float64": max_query * scpm_asset.Z_DIM * 8,
        "support_metric_features_float64": max_support * scpm_asset.Z_DIM * 8,
        "pairwise_support_scores_float64": max_query * max_support * 8,
        "class_logits_float64": max_query * base.g0.EXPECTED_CLASSES * 8,
        "runtime_state_numeric_bytes_max": max_state,
    }
    resource_peak = sum(resource_components.values())
    resource_budget = 1024 * 1024
    execution: dict[str, Any] = {
        "K_values": list(base.g0.K_VALUES),
        "fold_count": base.g0.EXPECTED_FOLDS,
        "query_count_per_k": base.g0.EXPECTED_ROWS,
        "common_query_order_root_sha256": query_root,
        "tap_snapshot_root_sha256": snapshot.tap_snapshot_root_sha256,
        "geometry": geometry,
        "argmax_changed_count_by_k": changed,
        "argmax_changed_count": sum(changed.values()),
        "zero_changed_k_values": zero_k,
        "functional_gate_status": (
            "G0_PASS_PROCEED_G1" if not zero_k else "REJECT_REVISION_NO_FUNCTION"
        ),
        "functional_gate_pass": not zero_k,
        "per_k": per_k,
        "resource_summary": {
            "sealed_prior_numeric_bytes": 12,
            "rank": scpm_asset.SCPM_RANK,
            "feature_dimension": scpm_asset.Z_DIM,
            "query_projection_rewrite_mac_estimate": 960,
            "incremental_numeric_array_components_bytes": resource_components,
            "incremental_numeric_array_peak_estimate_bytes": resource_peak,
            "numeric_array_budget_bytes": resource_budget,
            "resource_budget_exceeded": resource_peak > resource_budget,
            "parameter_scan_count": 0,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "process_rss_measured": False,
        },
    }
    execution["core_execution_root_sha256"] = base.g0._sha256(execution)
    return execution


def run_one_shot(
    *,
    archive_path: Path,
    expected_archive_sha256: str,
    registered_classes: Sequence[str],
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    if type(run_id) is not str or not run_id or len(run_id.encode("utf-8")) > 160:
        raise D110SCPMG0Error("run ID must be a short non-empty string")
    archive_bytes = base._read_pinned_archive(
        archive_path, expected_sha256=expected_archive_sha256
    )
    archive_sha256 = base._sha256_bytes(archive_bytes)
    rows = base._load_rows(archive_bytes, archive_sha256=archive_sha256)
    execution = _real_archive_execution(
        rows, registered_classes=registered_classes
    )
    result = {
        "schema": ONE_SHOT_SCHEMA,
        "status": ONE_SHOT_STATUS,
        "candidate_id": CANDIDATE_ID,
        "run_id": run_id,
        "archive_sha256": archive_sha256,
        "archive_member_names": list(base.TAP_MEMBERS),
        "row_count": base.g0.EXPECTED_ROWS,
        "real_archive_g0_executed": True,
        "g0_decision_consumption_allowed": True,
        "g1_entry_allowed": execution["functional_gate_pass"],
        "formal_performance_claim": False,
        "performance_metrics_emitted": False,
        "query_label_read_for_scoring": False,
        "metric_definitions": {
            "feature": "same-query D110 predictive-metric feature versus baseline normalized z_id",
            "neighbor": "nearest actual support ID signature under each method metric",
            "margin": "class-score top1-minus-top2 compared by binary64 bits",
            "argmax": "same-query D110 versus frozen qKNN class decision",
        },
        **execution,
    }
    base._assert_no_performance_fields(result)
    base._write_new_output(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--registered-class", required=True, action="append")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.name != "posix":
        raise D110SCPMG0Error("the one-shot command-line entry is POSIX-only")
    result = run_one_shot(
        archive_path=args.archive,
        expected_archive_sha256=args.archive_sha256,
        registered_classes=args.registered_class,
        run_id=args.run_id,
        output_path=args.output,
    )
    print(base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
