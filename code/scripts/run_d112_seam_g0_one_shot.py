#!/usr/bin/env python3
"""Run the D112 SEAM-qKNN no-truth functional check on the pinned 588-row tap."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import run_d111_r2_g0_one_shot as d111  # noqa: E402
from cvsrffi.stage2_d112_g0_source_bundle import (  # noqa: E402
    build_d112_g0_source_bundle,
)
from cvsrffi.stage2_d112_seam_qknn import (  # noqa: E402
    fit_d112_seam_g0_state,
    predict_d112_seam,
    score_d112_seam_logits,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    build_typed_zid_support_bank,
    decode_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
)


base = d111.base
SCHEMA = "cvs.phase1.d112.seam_g0.one_shot.v1"
STATUS = "REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE"
CANDIDATE_ID = "D112_SEAM_QKNN"


class D112G0Error(ValueError):
    """Raised when the D112 one-shot path cannot preserve its frozen semantics."""


def _effective_anchors(state: Any | None, bank: Any) -> np.ndarray:
    support = normalize_zid_rows(decode_zid_support_bank(bank).astype(np.float32)).astype(
        np.float64
    )
    result = np.zeros((len(bank.classes), base.g0.Z_DIM), dtype=np.float64)
    for class_index in range(len(bank.classes)):
        local = support[bank.class_indices_int16 == class_index]
        mean = np.mean(local, axis=0)
        norm = float(np.linalg.norm(mean))
        if len(local) != bank.active_k or not np.isfinite(mean).all() or norm <= 1.0e-12:
            raise D112G0Error("D112 G0 support prototype is invalid")
        result[class_index] = mean / norm
    if state is not None:
        for class_index in state.old_class_indices:
            if bool(state.information_valid[class_index]):
                result[class_index] = np.asarray(state.anchors[class_index], dtype=np.float64)
    return result


def _fold_audit(
    snapshot: Any,
    fold: Any,
    *,
    active_k: int,
    predecessor_lock: Any,
    registry: tuple[str, ...],
    bundle: Any,
) -> dict[str, Any]:
    inputs = base._fold_inputs(snapshot, fold, active_k=active_k, registry=registry)
    baseline_logits, _kernels, support_ids, baseline_features = base._baseline_trace(
        inputs, registry=registry, predecessor_lock=predecessor_lock
    )
    bank = build_typed_zid_support_bank(
        np.asarray(inputs["support_plus"], dtype=np.float32),
        tuple(inputs["support_labels"]),
        registry,
        config=predecessor_lock,
    )
    direct_baseline = base.g0.score_zid_student_t_logits(
        bank,
        np.asarray(inputs["query_plus"], dtype=np.float32),
        metric=identity_shared_psd_metric(config=predecessor_lock),
    )
    if not np.array_equal(baseline_logits, direct_baseline):
        raise D112G0Error("D112 G0 baseline qKNN path drift")
    state = fit_d112_seam_g0_state(bundle, bank)
    query = np.asarray(inputs["query_plus"], dtype=np.float32)
    candidate_logits = score_d112_seam_logits(state, bank, query)
    baseline_argmax = base.g0._unique_argmax(baseline_logits, registry)
    candidate_argmax = predict_d112_seam(state, bank, query)
    if candidate_argmax != base.g0._unique_argmax(candidate_logits, registry):
        raise D112G0Error("D112 G0 prediction/logit argmax drift")

    query_root = fold.query_root_sha256
    baseline_anchor = _effective_anchors(None, bank)
    candidate_anchor = _effective_anchors(state, bank)
    anchor = base._array_changed_metric(
        metric="effective_seam_anchor",
        query_root_sha256=query_root,
        baseline=baseline_anchor.view(np.uint64),
        candidate=candidate_anchor.view(np.uint64),
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
    support_root = base.g0._sha256(
        {
            "schema": SCHEMA + ".fixed_support_identity.v1",
            "support_ids_root_sha256": base.g0._sha256(list(support_ids)),
            "query_root_sha256": query_root,
        }
    )
    payload = {
        "query_root_sha256": query_root,
        "baseline_feature_root_sha256": base.g0._array_root(baseline_features),
        "candidate_feature_root_sha256": base.g0._array_root(baseline_features),
        "feature_changed_count": 0,
        "baseline_anchor_root_sha256": base.g0._array_root(baseline_anchor),
        "candidate_anchor_root_sha256": base.g0._array_root(candidate_anchor),
        "anchor_changed_count": int(anchor["changed_count"]),
        "anchor_changed_bitmap_root_sha256": anchor["changed_bitmap_root_sha256"],
        "baseline_score_root_sha256": base.g0._array_root(np.asarray(baseline_logits, dtype=np.float32)),
        "candidate_score_root_sha256": base.g0._array_root(np.asarray(candidate_logits, dtype=np.float32)),
        "score_changed_count": int(score["changed_count"]),
        "score_changed_bitmap_root_sha256": score["changed_bitmap_root_sha256"],
        "baseline_margin_root_sha256": base.g0._array_root(baseline_margin),
        "candidate_margin_root_sha256": base.g0._array_root(candidate_margin),
        "margin_changed_count": int(margin["changed_count"]),
        "margin_changed_bitmap_root_sha256": margin["changed_bitmap_root_sha256"],
        "baseline_argmax_root_sha256": base.g0._sha256(list(baseline_argmax)),
        "candidate_argmax_root_sha256": base.g0._sha256(list(candidate_argmax)),
        "argmax_changed_count": int(argmax["changed_count"]),
        "argmax_changed_bitmap_root_sha256": argmax["changed_bitmap_root_sha256"],
        "support_identity_root_sha256": support_root,
        "positive_rho_count": int(np.sum(np.asarray(state.rho) > 0.0)),
        "information_valid_count": int(np.sum(np.asarray(state.information_valid))),
        "donor_valid_count": int(np.sum(np.asarray(state.donor_valid))),
        "positive_anchor_shift_count": int(np.sum(np.asarray(state.anchor_shift_l2) > 0.0)),
        "max_anchor_shift_l2": float(np.max(state.anchor_shift_l2)),
        "max_rho": float(np.max(state.rho)),
        "runtime_state_numeric_bytes": int(state.resource_receipt["persistent_numeric_bytes"]),
        "global_bundle_valid": bool(state.global_bundle_valid),
    }
    payload["fold_mechanical_audit_root_sha256"] = base.g0._sha256(payload)
    return payload


def _aggregate(active_k: int, audits: Sequence[Mapping[str, Any]], query_root: str) -> dict[str, Any]:
    if len(audits) != base.g0.EXPECTED_FOLDS:
        raise D112G0Error("D112 G0 fold count drift")
    result: dict[str, Any] = {
        "K": active_k,
        "fold_count": len(audits),
        "query_count": base.g0.EXPECTED_ROWS,
        "query_ids_root_sha256": query_root,
    }
    for name in ("feature", "anchor", "score", "margin", "argmax"):
        result[f"{name}_changed_count"] = sum(
            int(item[f"{name}_changed_count"]) for item in audits
        )
    for name in (
        "positive_rho_count",
        "information_valid_count",
        "donor_valid_count",
        "positive_anchor_shift_count",
    ):
        result[name] = sum(int(item[name]) for item in audits)
    result["max_anchor_shift_l2"] = max(float(item["max_anchor_shift_l2"]) for item in audits)
    result["max_rho"] = max(float(item["max_rho"]) for item in audits)
    result["runtime_state_numeric_bytes_max"] = max(
        int(item["runtime_state_numeric_bytes"]) for item in audits
    )
    result["all_folds_global_bundle_valid"] = all(
        bool(item["global_bundle_valid"]) for item in audits
    )
    result["fold_mechanical_audits_root_sha256"] = base.g0._sha256(
        [item["fold_mechanical_audit_root_sha256"] for item in audits]
    )
    result["functional_nonzero"] = bool(
        result["positive_rho_count"] > 0
        and result["anchor_changed_count"] > 0
        and (result["score_changed_count"] > 0 or result["margin_changed_count"] > 0)
    )
    result["per_k_execution_root_sha256"] = base.g0._sha256(result)
    return result


def _decision(per_k: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    present = [int(item["K"]) for item in per_k if item["functional_nonzero"] is True]
    all_zero = all(
        int(item[name]) == 0
        for item in per_k
        for name in ("positive_rho_count", "anchor_changed_count", "score_changed_count", "margin_changed_count")
    )
    if present:
        status = "G0_FUNCTION_PRESENT_PROCEED_SINGLE_G1"
    elif all_zero:
        status = "REJECT_NO_FUNCTION_STRUCTURAL_ALL_K_ZERO"
    else:
        status = "G0_INCONCLUSIVE_PARTIAL_FUNCTION_REVIEW_IMPLEMENTATION"
    return {
        "functional_nonzero_k_values": present,
        "all_k_structurally_zero": all_zero,
        "functional_status": status,
        "g1_entry_allowed": bool(present),
    }


def run_one_shot(
    *,
    archive_path: Path,
    receipt_path: Path,
    expected_archive_sha256: str,
    checkpoint_sha256: str,
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    if type(run_id) is not str or not run_id or len(run_id.encode("utf-8")) > 160:
        raise D112G0Error("run ID must be a short non-empty string")
    archive_bytes = base._read_pinned_archive(
        archive_path, expected_sha256=expected_archive_sha256
    )
    archive_sha256 = base._sha256_bytes(archive_bytes)
    bundle = build_d112_g0_source_bundle(
        archive_path,
        receipt_path=receipt_path,
        checkpoint_sha256=checkpoint_sha256,
        expected_tap_sha256=expected_archive_sha256,
    )
    rows = base._load_rows(archive_bytes, archive_sha256=archive_sha256)
    registry = base.g0._canonical_registry(tuple(bundle.class_registry))
    snapshot = base.g0._snapshot_from_rows(
        rows,
        tap_receipt_sha256=base._sha256_bytes(base.g0._canonical_bytes(rows.receipt)),
    )
    locks = base._validate_one_shot_locks(
        base._predecessor_locks(rows), tap_receipt_sha256=snapshot.tap_receipt_sha256
    )
    plan = base.g0._build_fold_plan(snapshot)
    query_order = tuple(query for fold in plan for query in fold.query_ids)
    if len(query_order) != base.g0.EXPECTED_ROWS or len(set(query_order)) != base.g0.EXPECTED_ROWS:
        raise D112G0Error("D112 G0 common query closure drift")
    query_root = base.g0._sha256(list(query_order))
    per_k = []
    for active_k, lock in zip(base.g0.K_VALUES, locks, strict=True):
        audits = [
            _fold_audit(
                snapshot,
                fold,
                active_k=active_k,
                predecessor_lock=lock,
                registry=registry,
                bundle=bundle,
            )
            for fold in plan
        ]
        per_k.append(_aggregate(active_k, audits, query_root))
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "candidate_id": CANDIDATE_ID,
        "run_id": run_id,
        "archive_sha256": archive_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "bundle_content_root_sha256": bundle.manifest["content_root_sha256"],
        "source_aggregate_sha256": bundle.manifest["source_aggregate_sha256"],
        "global_bundle_valid": bundle.manifest["global_bundle_valid"],
        "global_invalid_reason": bundle.manifest["global_invalid_reason"],
        "row_count": base.g0.EXPECTED_ROWS,
        "fold_count": base.g0.EXPECTED_FOLDS,
        "K_values": list(base.g0.K_VALUES),
        "query_count_per_k": base.g0.EXPECTED_ROWS,
        "common_query_order_root_sha256": query_root,
        "real_archive_g0_executed": True,
        "formal_performance_claim": False,
        "performance_metrics_emitted": False,
        "query_label_read_for_scoring": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "parameter_scan_count": 0,
        "per_k": per_k,
        **_decision(per_k),
    }
    result["execution_root_sha256"] = base.g0._sha256(result)
    base._assert_no_performance_fields(result)
    base._write_new_output(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_one_shot(
        archive_path=args.archive,
        receipt_path=args.receipt,
        expected_archive_sha256=args.archive_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        run_id=args.run_id,
        output_path=args.output,
    )
    print(base._canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
