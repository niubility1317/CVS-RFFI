#!/usr/bin/env python3
"""Build the complete machine-readable M2.6 screen or full125 analysis."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

from cvsrffi.stage2_m24_safe_residual import D1
from scripts import summarize_m24_d1_refit_full125 as shared


M26_DIAGNOSTIC_METRICS = (
    "selected_strength",
    "fallback_to_zero",
    "gated_query_fraction",
    "adjusted_query_fraction",
    "max_logit_abs_delta",
    "identity_reliability",
    "envelope_reliability",
    "geometry_reliability",
    "identity_shift_norm",
    "envelope_shift_norm",
    "geometry_shift_norm",
    "identity_loo_gain_mean",
    "envelope_loo_gain_mean",
    "geometry_loo_gain_mean",
    "identity_loo_positive_fraction",
    "envelope_loo_positive_fraction",
    "geometry_loo_positive_fraction",
    "before_after_domain_digest_equal",
)


def _finite_mean(values: Iterable[Any]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else 0.0


def _positive_fraction(values: Iterable[Any]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(value > 0.0 for value in clean) / len(clean) if clean else 0.0


def _build_m26_diagnostics(matrix: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    reasons: Counter[tuple[str, str]] = Counter()
    for entry in matrix["entries"]:
        arm = str(entry["arm"])
        if arm == D1:
            continue
        receipt = json.loads(Path(entry["receipt_path"]).read_text(encoding="utf-8-sig"))
        condition = f"K{int(entry['k_shot'])}_new{int(entry['new_class_count'])}"
        for scene, audit in sorted(receipt["scenario_audit"].items()):
            domain = audit["domain_state"]
            application = audit["query_application"]
            query_count = int(application["query_count"])
            adjusted_count = int(application["adjusted_query_count"])
            before = audit.get("before_registration_fit", {})
            reason = str(audit["fallback_reason"])
            reasons[(arm, reason)] += 1
            records.append(
                {
                    "arm": arm,
                    "receiver": str(entry["receiver"]),
                    "method_seed": int(entry["method_seed"]),
                    "condition": condition,
                    "scene": str(scene),
                    "query_count": query_count,
                    "selected_strength": float(audit["selected_strength"]),
                    "fallback_to_zero": float(audit["selected_strength"] == 0.0),
                    "gated_query_fraction": float(application["gated_query_fraction"]),
                    "adjusted_query_fraction": adjusted_count / query_count if query_count else 0.0,
                    "max_logit_abs_delta": float(application["max_logit_abs_delta"]),
                    "identity_reliability": float(domain["identity_reliability"]),
                    "envelope_reliability": float(domain["envelope_reliability"]),
                    "geometry_reliability": float(domain["geometry_reliability"]),
                    "identity_shift_norm": float(domain["identity_shift_norm"]),
                    "envelope_shift_norm": float(domain["envelope_shift_norm"]),
                    "geometry_shift_norm": float(domain["geometry_shift_norm"]),
                    "identity_loo_gain_mean": _finite_mean(domain["identity_loo_gain"]),
                    "envelope_loo_gain_mean": _finite_mean(domain["envelope_loo_gain"]),
                    "geometry_loo_gain_mean": _finite_mean(domain["geometry_loo_gain"]),
                    "identity_loo_positive_fraction": _positive_fraction(domain["identity_loo_gain"]),
                    "envelope_loo_positive_fraction": _positive_fraction(domain["envelope_loo_gain"]),
                    "geometry_loo_positive_fraction": _positive_fraction(domain["geometry_loo_gain"]),
                    "before_after_domain_digest_equal": float(
                        before.get("domain_state_digest") == audit.get("domain_state_digest")
                    ),
                    "fallback_reason": reason,
                }
            )
    if not records:
        raise ValueError("M2.6 matrix contains no candidate diagnostics")
    grouped = lambda keys: shared._group_metric_rows(
        records, keys, M26_DIAGNOSTIC_METRICS, weight_key="query_count"
    )
    return {
        "metric_semantics": {
            "shift_norm": "norm of source-anchor to target-old-support shared shift",
            "reliability": "old-class leave-one-class-out transport reliability in [0,1]",
            "gated_query_fraction": "truth-blind fraction with B0 top-2 margin <= 0.10",
            "adjusted_query_fraction": "fraction receiving a nonzero bounded residual",
            "before_after_domain_digest_equal": "same old-support domain state before and after registration",
        },
        "overall": grouped(("arm",)),
        "condition": grouped(("arm", "condition")),
        "receiver": grouped(("arm", "receiver")),
        "seed": grouped(("arm", "method_seed")),
        "scene": grouped(("arm", "scene")),
        "fallback_reason_counts": [
            {"arm": arm, "fallback_reason": reason, "scenario_fit_count": count}
            for (arm, reason), count in sorted(reasons.items())
        ],
        "rows": records,
    }


def _write_summary_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _apply_m26_evidence_boundary(
    result: dict[str, Any], matrix: dict[str, Any]
) -> None:
    matrix_kind = str(matrix.get("matrix_kind", ""))
    identity_count = int(matrix["paired_input_identity_count"])
    if matrix_kind == "screen":
        boundary = (
            f"Same-row {identity_count}-identity screening evidence under p2_min_v1; "
            "not full-125 confirmation, Phase3, or deployment evidence."
        )
    elif matrix_kind == "full125":
        boundary = (
            "Same-row full-125 Stage2-C evidence under p2_min_v1; "
            "not Phase3 or deployment evidence."
        )
    else:
        raise ValueError(f"unsupported M2.6 matrix_kind: {matrix_kind!r}")
    result["matrix"]["matrix_kind"] = matrix_kind
    result["evidence_boundary"] = boundary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prediction_root = Path(args.prediction_root)
    matrix = json.loads((prediction_root / "matrix_index.json").read_text(encoding="utf-8-sig"))
    shared.ARMS = tuple(str(item) for item in matrix["arms"])
    shared.REFERENCE_ARM = D1
    shared.PARITY_ARM = None
    shared.EXPECTED_INPUT_IDENTITIES = int(matrix["paired_input_identity_count"])
    shared.SUMMARY_SCHEMA = "cvs.erbt_idr.m26.td_src256.results_summary.v1"
    shared.SUMMARY_VERDICT = "M26_TD_SRC256_MATRIX_MEASURED"
    result = shared.build_summary(prediction_root, Path(args.score_root))
    _apply_m26_evidence_boundary(result, matrix)
    result["m26_diagnostics"] = _build_m26_diagnostics(matrix)
    _write_summary_exclusive(Path(args.output), result)
    print(json.dumps({"status": result["status"], "row_count": result["matrix"]["row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
