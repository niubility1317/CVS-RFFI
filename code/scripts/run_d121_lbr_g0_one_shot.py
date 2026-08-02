#!/usr/bin/env python3
"""Run D121 LBR M_HEAD on the pinned 588-row tap without performance output."""

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
from cvsrffi.stage2_d121_lbr_qknn import (  # noqa: E402
    audit_lbr_qknn_state,
    build_lbr_qknn_state,
    score_lbr_qknn_trace,
    unique_lbr_argmax,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.phase1.d121.lbr_qknn_g0.one_shot.v1"
STATUS = "REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE"
CANDIDATE_ID = "D121_LBR_QKNN_M_HEAD"


class D121G0Error(ValueError):
    """Raised when the fixed D121 588-row functional path drifts."""


def functional_gate_from_argmax_counts(counts: Mapping[int | str, int]) -> dict[str, Any]:
    """Apply D121's sole G0 decision: each frozen K must alter an argmax."""

    normalized: dict[str, int] = {}
    for active_k in base.g0.K_VALUES:
        raw = counts.get(active_k, counts.get(str(active_k)))
        if type(raw) is not int or raw < 0:
            raise D121G0Error("G0 argmax counts must provide non-negative K1/K5/K10 values")
        normalized[str(active_k)] = raw
    zero = [active_k for active_k in base.g0.K_VALUES if normalized[str(active_k)] == 0]
    passed = not zero
    return {
        "argmax_changed_count_by_k": normalized,
        "zero_changed_k_values": zero,
        "functional_gate_status": "G0_PASS_PROCEED_G1" if passed else "REJECT_REVISION_NO_FUNCTION",
        "functional_gate_pass": passed,
    }


def _require_registered_classes(value: Sequence[str]) -> tuple[str, ...]:
    return base.g0._canonical_registry(base._require_registered_classes(value))


def _fold_audit(
    snapshot: Any,
    fold: Any,
    *,
    active_k: int,
    predecessor_lock: Any,
    registry: tuple[str, ...],
) -> dict[str, Any]:
    """Compare M0 and LBR for one fixed held cell without opening scores."""

    inputs = base._fold_inputs(snapshot, fold, active_k=active_k, registry=registry)
    baseline_logits, baseline_kernels, support_ids, _query_features = base._baseline_trace(
        inputs, registry=registry, predecessor_lock=predecessor_lock
    )
    bank = build_typed_zid_support_bank(
        np.asarray(inputs["support_plus"], dtype=np.float32),
        tuple(inputs["support_labels"]),
        registry,
        config=predecessor_lock,
    )
    metric = identity_shared_psd_metric(config=predecessor_lock)
    direct_baseline = score_zid_student_t_logits(
        bank, np.asarray(inputs["query_plus"], dtype=np.float32), metric=metric
    )
    if not np.array_equal(baseline_logits, direct_baseline):
        raise D121G0Error("D121 G0 baseline qKNN path drift")
    state = build_lbr_qknn_state(bank, support_ids, metric=metric)
    trace = score_lbr_qknn_trace(
        state, bank, np.asarray(inputs["query_plus"], dtype=np.float32), metric=metric
    )
    candidate_logits = trace.class_logits_fp32
    baseline_argmax = base.g0._unique_argmax(baseline_logits, registry)
    candidate_argmax = unique_lbr_argmax(candidate_logits, registry)
    query_root = fold.query_root_sha256

    support_kernel = base._array_changed_metric(
        metric="lbr_support_kernel_logit_binary64",
        query_root_sha256=query_root,
        baseline=np.asarray(baseline_kernels, dtype=np.float64).view(np.uint64),
        candidate=np.asarray(trace.lbr_support_logits_fp64, dtype=np.float64).view(np.uint64),
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
    audit = audit_lbr_qknn_state(state)
    payload = {
        "query_root_sha256": query_root,
        "lbr_rival_index_root_sha256": state.rival_index_root_sha256,
        "lbr_support_identity_root_sha256": state.support_identity_root_sha256,
        "rival_index_count": int(len(state.rival_indices_uint16)),
        "rival_non_identity_count": int(
            np.count_nonzero(
                state.rival_indices_uint16.astype(np.int64)
                != np.arange(len(state.rival_indices_uint16), dtype=np.int64)
            )
        ),
        "baseline_support_kernel_root_sha256": base.g0._array_root(
            np.asarray(baseline_kernels, dtype=np.float64)
        ),
        "candidate_support_kernel_root_sha256": base.g0._array_root(
            np.asarray(trace.lbr_support_logits_fp64, dtype=np.float64)
        ),
        "support_kernel_changed_count": int(support_kernel["changed_count"]),
        "support_kernel_changed_bitmap_root_sha256": support_kernel[
            "changed_bitmap_root_sha256"
        ],
        "baseline_score_root_sha256": base.g0._array_root(
            np.asarray(baseline_logits, dtype=np.float32)
        ),
        "candidate_score_root_sha256": base.g0._array_root(candidate_logits),
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
        "runtime_state_numeric_bytes": int(
            audit["resource_receipt"]["persistent_numeric_bytes"]
        ),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }
    payload["fold_mechanical_audit_root_sha256"] = base.g0._sha256(payload)
    return payload


def _aggregate_per_k(
    *, active_k: int, audits: Sequence[Mapping[str, Any]], query_root_sha256: str
) -> dict[str, Any]:
    if len(audits) != base.g0.EXPECTED_FOLDS:
        raise D121G0Error("D121 G0 fold audit count drift")
    result: dict[str, Any] = {
        "K": int(active_k),
        "fold_count": base.g0.EXPECTED_FOLDS,
        "query_count": base.g0.EXPECTED_ROWS,
        "query_ids_root_sha256": query_root_sha256,
        "fold_mechanical_audits_root_sha256": base.g0._sha256(
            [item["fold_mechanical_audit_root_sha256"] for item in audits]
        ),
        "lbr_rival_index_roots_root_sha256": base.g0._sha256(
            [item["lbr_rival_index_root_sha256"] for item in audits]
        ),
        "rival_index_count": sum(int(item["rival_index_count"]) for item in audits),
        "rival_non_identity_count": sum(
            int(item["rival_non_identity_count"]) for item in audits
        ),
        "runtime_state_numeric_bytes_max": max(
            int(item["runtime_state_numeric_bytes"]) for item in audits
        ),
    }
    for metric in ("support_kernel", "score", "margin", "argmax"):
        result[f"{metric}_changed_count"] = sum(
            int(item[f"{metric}_changed_count"]) for item in audits
        )
        for side in ("baseline", "candidate"):
            result[f"{side}_{metric}_root_sha256"] = base.g0._sha256(
                [item[f"{side}_{metric}_root_sha256"] for item in audits]
            )
        result[f"{metric}_changed_bitmap_roots_root_sha256"] = base.g0._sha256(
            [item[f"{metric}_changed_bitmap_root_sha256"] for item in audits]
        )
    result["per_k_execution_root_sha256"] = base.g0._sha256(result)
    return result


def _resource_summary(per_k: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        "feature_dimension": base.g0.Z_DIM,
        "rival_index_dtype": "uint16",
        "runtime_state_numeric_bytes_max": max(
            int(item["runtime_state_numeric_bytes_max"]) for item in per_k
        ),
        "design_max_persistent_numeric_bytes_at_n260": 520,
        "parameter_scan_count": 0,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "process_rss_measured": False,
    }
    result["resource_summary_root_sha256"] = base.g0._sha256(result)
    return result


def _real_archive_execution(
    rows: Any, *, registered_classes: Sequence[str]
) -> dict[str, Any]:
    registry = _require_registered_classes(registered_classes)
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
        raise D121G0Error("D121 G0 common query closure drift")
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
            )
            for fold in plan
        ]
        per_k.append(
            _aggregate_per_k(
                active_k=active_k, audits=audits, query_root_sha256=query_root
            )
        )
    gate = functional_gate_from_argmax_counts(
        {int(item["K"]): int(item["argmax_changed_count"]) for item in per_k}
    )
    result = {
        "K_values": list(base.g0.K_VALUES),
        "fold_count": base.g0.EXPECTED_FOLDS,
        "query_count_per_k": base.g0.EXPECTED_ROWS,
        "common_query_order_root_sha256": query_root,
        "tap_snapshot_root_sha256": snapshot.tap_snapshot_root_sha256,
        "support_kernel_changed_count_by_k": {
            str(item["K"]): int(item["support_kernel_changed_count"]) for item in per_k
        },
        "margin_changed_count_by_k": {
            str(item["K"]): int(item["margin_changed_count"]) for item in per_k
        },
        **gate,
        "per_k": per_k,
        "resource_summary": _resource_summary(per_k),
    }
    result["core_execution_root_sha256"] = base.g0._sha256(result)
    return result


def run_one_shot(
    *,
    archive_path: Path,
    expected_archive_sha256: str,
    registered_classes: Sequence[str],
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    """Execute the fixed LBR G0 functional check and write a new artifact."""

    if type(run_id) is not str or not run_id or len(run_id.encode("utf-8")) > 160:
        raise D121G0Error("run ID must be a short non-empty string")
    archive_bytes = base._read_pinned_archive(
        archive_path, expected_sha256=expected_archive_sha256
    )
    archive_sha256 = base._sha256_bytes(archive_bytes)
    rows = base._load_rows(archive_bytes, archive_sha256=archive_sha256)
    execution = _real_archive_execution(rows, registered_classes=registered_classes)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
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
            "rival_index": "support-only closest foreign-class edge with physical/content hash tie closure",
            "support_kernel": "single-hop stable LBR support-logit correction versus frozen M0 kernel",
            "margin": "same-query top1-minus-top2 class-score margin compared by binary64 bits",
            "argmax": "same-query D121 LBR versus frozen M0 class decision",
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
        raise D121G0Error("the one-shot command-line entry is POSIX-only")
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
