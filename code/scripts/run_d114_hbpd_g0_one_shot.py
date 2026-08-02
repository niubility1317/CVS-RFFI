#!/usr/bin/env python3
"""Run D114 HBPD-qKNN on the pinned 588-row tap without opening truth."""

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
from cvsrffi.stage2_d114_g0_source_bundle import (  # noqa: E402
    build_d114_g0_source_bundle,
)
from cvsrffi.stage2_d114_hbpd_qknn import (  # noqa: E402
    fit_d114_state,
    score_d114_hbpd_logits,
)
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
)


base = d111.base
SCHEMA = "cvs.phase1.d114.hbpd_g0.one_shot.v1"
STATUS = "REAL_ARCHIVE_G0_EXECUTED_NON_FORMAL_FUNCTIONAL_EVIDENCE"
CANDIDATE_ID = "D114_HBPD_QKNN_M_DA"


class D114G0Error(ValueError):
    pass


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
    query = np.asarray(inputs["query_plus"], dtype=np.float32)
    direct_baseline = base.g0.score_zid_student_t_logits(
        bank, query, metric=identity_shared_psd_metric(config=predecessor_lock)
    )
    if not np.array_equal(baseline_logits, direct_baseline):
        raise D114G0Error("D114 G0 baseline path drift")
    state = fit_d114_state(bundle, bank)
    candidate_logits = score_d114_hbpd_logits(state, bank, query)
    baseline_argmax = base.g0._unique_argmax(baseline_logits, registry)
    candidate_argmax = base.g0._unique_argmax(candidate_logits, registry)
    query_root = fold.query_root_sha256
    bandwidth = base._array_changed_metric(
        metric="class_predictive_bandwidth_binary64",
        query_root_sha256=query_root,
        baseline=np.asarray(bank.class_scales_fp16, dtype=np.float64).view(np.uint64),
        candidate=np.asarray(state.predictive_bandwidth, dtype=np.float64).view(np.uint64),
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
    payload = {
        "query_root_sha256": query_root,
        "support_identity_root_sha256": base.g0._sha256(
            {
                "support_ids_root_sha256": base.g0._sha256(list(support_ids)),
                "query_root_sha256": query_root,
            }
        ),
        "baseline_feature_root_sha256": base.g0._array_root(
            np.asarray(baseline_features, dtype=np.float32)
        ),
        "candidate_feature_root_sha256": base.g0._array_root(
            np.asarray(baseline_features, dtype=np.float32)
        ),
        "feature_changed_count": 0,
        "bandwidth_changed_count": int(bandwidth["changed_count"]),
        "bandwidth_changed_bitmap_root_sha256": bandwidth["changed_bitmap_root_sha256"],
        "score_changed_count": int(score["changed_count"]),
        "score_changed_bitmap_root_sha256": score["changed_bitmap_root_sha256"],
        "margin_changed_count": int(margin["changed_count"]),
        "margin_changed_bitmap_root_sha256": margin["changed_bitmap_root_sha256"],
        "argmax_changed_count": int(argmax["changed_count"]),
        "argmax_changed_bitmap_root_sha256": argmax["changed_bitmap_root_sha256"],
        "bandwidth_min": float(np.min(state.predictive_bandwidth)),
        "bandwidth_max": float(np.max(state.predictive_bandwidth)),
        "bandwidth_unique_count": int(len(np.unique(state.predictive_bandwidth))),
        "runtime_state_numeric_bytes": int(
            state.resource_receipt["persistent_numeric_bytes"]
        ),
        "query_rows_used_for_fit": 0,
        "truth_role_quota_inputs": 0,
    }
    payload["fold_mechanical_audit_root_sha256"] = base.g0._sha256(payload)
    return payload


def _aggregate(
    active_k: int, audits: Sequence[Mapping[str, Any]], query_root: str
) -> dict[str, Any]:
    if len(audits) != base.g0.EXPECTED_FOLDS:
        raise D114G0Error("D114 G0 fold count drift")
    result: dict[str, Any] = {
        "K": active_k,
        "fold_count": len(audits),
        "query_count": base.g0.EXPECTED_ROWS,
        "query_ids_root_sha256": query_root,
        "feature_changed_count": 0,
    }
    for name in ("bandwidth", "score", "margin", "argmax"):
        result[f"{name}_changed_count"] = sum(
            int(item[f"{name}_changed_count"]) for item in audits
        )
    result["bandwidth_min"] = min(float(item["bandwidth_min"]) for item in audits)
    result["bandwidth_max"] = max(float(item["bandwidth_max"]) for item in audits)
    result["bandwidth_unique_count_min"] = min(
        int(item["bandwidth_unique_count"]) for item in audits
    )
    result["runtime_state_numeric_bytes_max"] = max(
        int(item["runtime_state_numeric_bytes"]) for item in audits
    )
    result["fold_mechanical_audits_root_sha256"] = base.g0._sha256(
        [item["fold_mechanical_audit_root_sha256"] for item in audits]
    )
    result["functional_nonzero"] = bool(
        result["bandwidth_changed_count"] > 0
        and result["score_changed_count"] > 0
        and result["margin_changed_count"] > 0
        and result["argmax_changed_count"] > 0
    )
    result["per_k_execution_root_sha256"] = base.g0._sha256(result)
    return result


def _decision(per_k: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    zero = [int(item["K"]) for item in per_k if int(item["argmax_changed_count"]) == 0]
    return {
        "zero_argmax_k_values": zero,
        "functional_status": (
            "REJECT_REVISION_NO_FUNCTION"
            if zero
            else "G0_ALL_K_ARGMAX_NONZERO_PROCEED_G1"
        ),
        "g1_entry_allowed": not zero,
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
        raise D114G0Error("run ID must be a short non-empty string")
    archive_bytes = base._read_pinned_archive(
        archive_path, expected_sha256=expected_archive_sha256
    )
    archive_sha256 = base._sha256_bytes(archive_bytes)
    rows = base._load_rows(archive_bytes, archive_sha256=archive_sha256)
    snapshot = base.g0._snapshot_from_rows(
        rows,
        tap_receipt_sha256=base._sha256_bytes(base.g0._canonical_bytes(rows.receipt)),
    )
    locks = base._validate_one_shot_locks(
        base._predecessor_locks(rows), tap_receipt_sha256=snapshot.tap_receipt_sha256
    )
    bundle = build_d114_g0_source_bundle(
        archive_path,
        receipt_path=receipt_path,
        checkpoint_sha256=checkpoint_sha256,
        expected_tap_sha256=expected_archive_sha256,
        allowed_config_lock_digests=tuple(lock.lock_digest for lock in locks),
    )
    registry = base.g0._canonical_registry(tuple(bundle.class_registry))
    plan = base.g0._build_fold_plan(snapshot)
    query_order = tuple(query for fold in plan for query in fold.query_ids)
    if len(query_order) != base.g0.EXPECTED_ROWS or len(set(query_order)) != base.g0.EXPECTED_ROWS:
        raise D114G0Error("D114 G0 common query closure drift")
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
        "bundle_content_sha256": bundle.content_sha256,
        "source_aggregate_sha256": bundle.source_aggregate_sha256,
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
