#!/usr/bin/env python3
"""Run the D111-r2 G0 functional check from a pinned 588-row strict tap.

This is a deliberately thin, non-formal entry.  It reuses D106's fixed
leave-cell-out folds and M0 qKNN path, then compares only D111 anchor, score,
margin and argmax mechanics.  It neither reads nor emits performance labels or
performance metrics.
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
from cvsrffi.stage2_d111_g0_source_bundle import (  # noqa: E402
    COMPONENT_STATE,
    load_d111_g0_source_bundle,
)
from cvsrffi.stage2_d111_loo_gat_score import (  # noqa: E402
    fit_d111_loo_gat_g0_state,
    predict_d111_loo_gat,
    score_d111_loo_gat_logits,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    build_typed_zid_support_bank,
    decode_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
)


ONE_SHOT_SCHEMA = "cvs.phase1.d111.r2_g0.one_shot.v1"
ONE_SHOT_STATUS = "REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE"
CANDIDATE_ID = "D111_R2_LOO_GAT"


class D111R2G0Error(ValueError):
    """Raised when the small D111 G0 path cannot preserve frozen semantics."""


def functional_gate_from_argmax_counts(counts: Mapping[int | str, int]) -> dict[str, Any]:
    """Apply the only G0 decision rule: every frozen K must change an argmax."""

    normalised: dict[str, int] = {}
    for active_k in base.g0.K_VALUES:
        raw = counts.get(active_k, counts.get(str(active_k)))
        if type(raw) is not int or raw < 0:
            raise D111R2G0Error("G0 argmax count mapping must provide non-negative K1/K5/K10 counts")
        normalised[str(active_k)] = raw
    if len(normalised) != len(base.g0.K_VALUES):
        raise D111R2G0Error("G0 argmax count mapping is incomplete")
    zero = [active_k for active_k in base.g0.K_VALUES if normalised[str(active_k)] == 0]
    passed = not zero
    return {
        "argmax_changed_count_by_k": normalised,
        "zero_changed_k_values": zero,
        "functional_gate_status": "G0_PASS_PROCEED_G1" if passed else "REJECT_REVISION_NO_FUNCTION",
        "functional_gate_pass": passed,
    }


def _require_registered_classes(value: Sequence[str], *, bundle_classes: tuple[str, ...]) -> tuple[str, ...]:
    classes = base.g0._canonical_registry(base._require_registered_classes(value))
    if classes != bundle_classes:
        raise D111R2G0Error("registered class order/content must equal the pinned D111 G0 aggregate")
    return classes


def _effective_anchors(state: Any, bank: Any) -> np.ndarray:
    """Return only anchors with nonzero mixture mass; zero-mass rows are M0."""

    support = normalize_zid_rows(decode_zid_support_bank(bank).astype(np.float32)).astype(np.float64)
    baseline = np.zeros((len(bank.classes), base.g0.Z_DIM), dtype=np.float64)
    for class_index in range(len(bank.classes)):
        local = support[bank.class_indices_int16 == class_index]
        if len(local) != bank.active_k:
            raise D111R2G0Error("D111 G0 support count drift")
        mean = np.mean(local, axis=0)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(mean).all() or norm <= 1.0e-12:
            raise D111R2G0Error("D111 G0 support mean is invalid")
        baseline[class_index] = mean / norm
    candidate = baseline.copy()
    for class_index in state.old_class_indices:
        if float(state.rho[class_index]) > 0.0:
            candidate[class_index] = np.asarray(state.anchors[class_index], dtype=np.float64)
    if not np.isfinite(candidate).all():
        raise D111R2G0Error("D111 G0 effective anchors became non-finite")
    return candidate


def _fold_audit(
    snapshot: Any,
    fold: Any,
    *,
    active_k: int,
    predecessor_lock: Any,
    registry: tuple[str, ...],
    bundle: Any,
) -> dict[str, Any]:
    """Compare one fixed fold without exposing labels or predictions."""

    inputs = base._fold_inputs(snapshot, fold, active_k=active_k, registry=registry)
    baseline_logits, _kernels, baseline_support_ids, baseline_features = base._baseline_trace(
        inputs, registry=registry, predecessor_lock=predecessor_lock
    )
    bank = build_typed_zid_support_bank(
        np.asarray(inputs["support_plus"], dtype=np.float32),
        tuple(inputs["support_labels"]),
        registry,
        config=predecessor_lock,
    )
    if not np.array_equal(
        baseline_logits,
        base.g0.score_zid_student_t_logits(
            bank,
            np.asarray(inputs["query_plus"], dtype=np.float32),
            metric=identity_shared_psd_metric(config=predecessor_lock),
        ),
    ):
        raise D111R2G0Error("D111 G0 baseline qKNN path drift")
    state = fit_d111_loo_gat_g0_state(bundle, bank)
    candidate_logits = score_d111_loo_gat_logits(
        state, bank, np.asarray(inputs["query_plus"], dtype=np.float32)
    )
    baseline_argmax = base.g0._unique_argmax(baseline_logits, registry)
    candidate_argmax = predict_d111_loo_gat(
        state, bank, np.asarray(inputs["query_plus"], dtype=np.float32)
    )
    direct_candidate = base.g0._unique_argmax(candidate_logits, registry)
    if candidate_argmax != direct_candidate:
        raise D111R2G0Error("D111 G0 prediction/logit argmax drift")
    query_root = fold.query_root_sha256
    baseline_anchor = _effective_anchors(
        type("BaselineState", (), {"old_class_indices": (), "rho": (), "anchors": ()})(), bank
    )
    candidate_anchor = _effective_anchors(state, bank)
    anchor = base._array_changed_metric(
        metric="effective_transport_anchor",
        query_root_sha256=query_root,
        baseline=baseline_anchor.view(np.uint64),
        candidate=candidate_anchor.view(np.uint64),
    )
    # D111 only adds a support-derived class anchor mixture.  It neither
    # rewrites support rows nor changes the canonical M0 support-neighbor
    # identity.  Bind that unchanged identity without exposing its IDs.
    support_neighbor_identity_root = base.g0._sha256(
        {
            "schema": ONE_SHOT_SCHEMA + ".fixed_support_neighbor_identity.v1",
            "support_ids_root_sha256": base.g0._sha256(list(baseline_support_ids)),
            "query_root_sha256": query_root,
        }
    )
    neighbor = base._changed_metric(
        metric="fixed_support_neighbor_identity",
        query_root_sha256=query_root,
        baseline=(support_neighbor_identity_root,) * len(baseline_features),
        candidate=(support_neighbor_identity_root,) * len(baseline_features),
    )
    score = base._array_changed_metric(
        metric="class_score_binary32",
        query_root_sha256=query_root,
        baseline=np.asarray(baseline_logits, dtype=np.float32).view(np.uint32),
        candidate=np.asarray(candidate_logits, dtype=np.float32).view(np.uint32),
    )
    baseline_margin = base._top_two_margins(np.asarray(baseline_logits, dtype=np.float64))
    candidate_margin = base._top_two_margins(np.asarray(candidate_logits, dtype=np.float64))
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
    # D111 changes only support-derived anchor density.  It does not rewrite a
    # query feature, so this count is intentionally and mechanically zero.
    feature = {"changed_count": 0, "changed_bitmap_root_sha256": base.g0._sha256({"schema": ONE_SHOT_SCHEMA, "metric": "query_feature_identity", "query_root_sha256": query_root, "bits": "0" * len(baseline_features)})}
    payload = {
        "query_root_sha256": query_root,
        "baseline_feature_root_sha256": base.g0._array_root(baseline_features),
        "candidate_feature_root_sha256": base.g0._array_root(baseline_features),
        "baseline_anchor_root_sha256": base.g0._array_root(baseline_anchor),
        "candidate_anchor_root_sha256": base.g0._array_root(candidate_anchor),
        "anchor_changed_count": anchor["changed_count"],
        "anchor_changed_bitmap_root_sha256": anchor["changed_bitmap_root_sha256"],
        "baseline_neighbor_root_sha256": support_neighbor_identity_root,
        "candidate_neighbor_root_sha256": support_neighbor_identity_root,
        "neighbor_changed_count": neighbor["changed_count"],
        "neighbor_changed_bitmap_root_sha256": neighbor["changed_bitmap_root_sha256"],
        "baseline_score_root_sha256": base.g0._array_root(np.asarray(baseline_logits, dtype=np.float32)),
        "candidate_score_root_sha256": base.g0._array_root(np.asarray(candidate_logits, dtype=np.float32)),
        "score_changed_count": score["changed_count"],
        "score_changed_bitmap_root_sha256": score["changed_bitmap_root_sha256"],
        "baseline_margin_root_sha256": base.g0._array_root(baseline_margin),
        "candidate_margin_root_sha256": base.g0._array_root(candidate_margin),
        "margin_changed_count": margin["changed_count"],
        "margin_changed_bitmap_root_sha256": margin["changed_bitmap_root_sha256"],
        "baseline_argmax_root_sha256": base.g0._sha256(list(baseline_argmax)),
        "candidate_argmax_root_sha256": base.g0._sha256(list(candidate_argmax)),
        "argmax_changed_count": argmax["changed_count"],
        "argmax_changed_bitmap_root_sha256": argmax["changed_bitmap_root_sha256"],
        "feature_changed_count": feature["changed_count"],
        "feature_changed_bitmap_root_sha256": feature["changed_bitmap_root_sha256"],
        "positive_anchor_mass_count": int(np.sum(np.asarray(state.rho) > 0.0)),
        "qualified_anchor_count": int(np.sum(np.asarray(state.qualified))),
        "runtime_state_numeric_bytes": int(state.resource_receipt["persistent_numeric_bytes"]),
    }
    payload["fold_mechanical_audit_root_sha256"] = base.g0._sha256(payload)
    return payload


def _aggregate_per_k(
    *, active_k: int, audits: Sequence[Mapping[str, Any]], query_root_sha256: str
) -> dict[str, Any]:
    if len(audits) != base.g0.EXPECTED_FOLDS:
        raise D111R2G0Error("D111 G0 fold audit count drift")
    result: dict[str, Any] = {
        "K": active_k,
        "fold_count": base.g0.EXPECTED_FOLDS,
        "query_count": base.g0.EXPECTED_ROWS,
        "query_ids_root_sha256": query_root_sha256,
        "fold_mechanical_audits_root_sha256": base.g0._sha256(
            [item["fold_mechanical_audit_root_sha256"] for item in audits]
        ),
    }
    for metric in ("feature", "anchor", "neighbor", "score", "margin", "argmax"):
        result[f"{metric}_changed_count"] = sum(int(item[f"{metric}_changed_count"]) for item in audits)
        for side in ("baseline", "candidate"):
            result[f"{side}_{metric}_root_sha256"] = base.g0._sha256(
                [item[f"{side}_{metric}_root_sha256"] for item in audits]
            )
        result[f"{metric}_changed_bitmap_roots_root_sha256"] = base.g0._sha256(
            [item[f"{metric}_changed_bitmap_root_sha256"] for item in audits]
        )
    result["positive_anchor_mass_count"] = sum(int(item["positive_anchor_mass_count"]) for item in audits)
    result["qualified_anchor_count"] = sum(int(item["qualified_anchor_count"]) for item in audits)
    result["runtime_state_numeric_bytes_max"] = max(int(item["runtime_state_numeric_bytes"]) for item in audits)
    result["per_k_execution_root_sha256"] = base.g0._sha256(result)
    return result


def _resource_summary(bundle: Any, per_k: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload_resource = dict(bundle.manifest["resource_receipt"])
    result = {
        "bundle_numeric_payload_bytes": int(payload_resource["numeric_payload_bytes"]),
        "runtime_state_numeric_bytes_max": max(int(item["runtime_state_numeric_bytes_max"]) for item in per_k),
        "feature_dimension": base.g0.Z_DIM,
        "rank": 3,
        "extra_query_anchor_macs_upper_bound": 6 * base.g0.Z_DIM,
        "parameter_scan_count": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "feature_changed_count": 0,
        "process_rss_measured": False,
    }
    result["resource_summary_root_sha256"] = base.g0._sha256(result)
    return result


def _real_archive_execution(rows: Any, *, registered_classes: Sequence[str], bundle: Any) -> dict[str, Any]:
    registry = _require_registered_classes(registered_classes, bundle_classes=tuple(bundle.class_registry))
    tap_receipt_sha256 = base._sha256_bytes(base.g0._canonical_bytes(rows.receipt))
    snapshot = base.g0._snapshot_from_rows(rows, tap_receipt_sha256=tap_receipt_sha256)
    locks = base._validate_one_shot_locks(
        base._predecessor_locks(rows), tap_receipt_sha256=snapshot.tap_receipt_sha256
    )
    plan = base.g0._build_fold_plan(snapshot)
    query_order = tuple(query for fold in plan for query in fold.query_ids)
    if len(query_order) != base.g0.EXPECTED_ROWS or len(set(query_order)) != base.g0.EXPECTED_ROWS:
        raise D111R2G0Error("D111 G0 common query closure drift")
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
                bundle=bundle,
            )
            for fold in plan
        ]
        per_k.append(_aggregate_per_k(active_k=active_k, audits=audits, query_root_sha256=query_root))
    gate = functional_gate_from_argmax_counts(
        {int(item["K"]): int(item["argmax_changed_count"]) for item in per_k}
    )
    result = {
        "K_values": list(base.g0.K_VALUES),
        "fold_count": base.g0.EXPECTED_FOLDS,
        "query_count_per_k": base.g0.EXPECTED_ROWS,
        "common_query_order_root_sha256": query_root,
        "tap_snapshot_root_sha256": snapshot.tap_snapshot_root_sha256,
        "feature_changed_count_by_k": {str(item["K"]): int(item["feature_changed_count"]) for item in per_k},
        "anchor_changed_count_by_k": {str(item["K"]): int(item["anchor_changed_count"]) for item in per_k},
        "neighbor_changed_count_by_k": {str(item["K"]): int(item["neighbor_changed_count"]) for item in per_k},
        "score_changed_count_by_k": {str(item["K"]): int(item["score_changed_count"]) for item in per_k},
        "margin_changed_count_by_k": {str(item["K"]): int(item["margin_changed_count"]) for item in per_k},
        **gate,
        "per_k": per_k,
        "resource_summary": _resource_summary(bundle, per_k),
    }
    result["core_execution_root_sha256"] = base.g0._sha256(result)
    return result


def run_one_shot(
    *,
    archive_path: Path,
    expected_archive_sha256: str,
    bundle_dir: Path,
    registered_classes: Sequence[str],
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run the one-shot D111 G0 closure and write a new non-performance result."""

    if type(run_id) is not str or not run_id or len(run_id.encode("utf-8")) > 160:
        raise D111R2G0Error("run ID must be a short non-empty string")
    archive_bytes = base._read_pinned_archive(archive_path, expected_sha256=expected_archive_sha256)
    archive_sha256 = base._sha256_bytes(archive_bytes)
    bundle = load_d111_g0_source_bundle(bundle_dir)
    if bundle.manifest.get("source_tap_sha256") != archive_sha256:
        raise D111R2G0Error("D111 G0 bundle does not bind the pinned strict tap")
    if bundle.manifest.get("effective_bundle_state") != COMPONENT_STATE:
        raise D111R2G0Error("D111 G0 bundle lifecycle marker drift")
    rows = base._load_rows(archive_bytes, archive_sha256=archive_sha256)
    execution = _real_archive_execution(rows, registered_classes=registered_classes, bundle=bundle)
    result = {
        "schema": ONE_SHOT_SCHEMA,
        "status": ONE_SHOT_STATUS,
        "candidate_id": CANDIDATE_ID,
        "run_id": run_id,
        "archive_sha256": archive_sha256,
        "bundle_payload_sha256": bundle.manifest["payload_sha256"],
        "row_count": base.g0.EXPECTED_ROWS,
        "real_archive_g0_executed": True,
        "g0_decision_consumption_allowed": True,
        "g1_entry_allowed": execution["functional_gate_pass"],
        "formal_performance_claim": False,
        "performance_metrics_emitted": False,
        "query_label_read_for_scoring": False,
        "metric_definitions": {
            "feature": "same-query normalized feature; D111 intentionally leaves it unchanged",
            "anchor": "effective nonzero-mass transported anchor versus the M0 support mean",
            "neighbor": "fixed canonical M0 support-neighbor identity; D111 intentionally leaves it unchanged",
            "score": "same-query class-score vector compared by binary32 bits",
            "margin": "same-query top1-minus-top2 class-score margin compared by binary64 bits",
            "argmax": "same-query D111 versus frozen M0 qKNN class decision",
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
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--registered-class", required=True, action="append")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.name != "posix":
        raise D111R2G0Error("the one-shot command-line entry is POSIX-only")
    result = run_one_shot(
        archive_path=args.archive,
        expected_archive_sha256=args.archive_sha256,
        bundle_dir=args.bundle,
        registered_classes=args.registered_class,
        run_id=args.run_id,
        output_path=args.output,
    )
    print(base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
