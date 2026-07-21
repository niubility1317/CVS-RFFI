#!/usr/bin/env python3
"""Summarize a fresh, paired D92 role-only Oracle 125 upper-bound run.

The input is a sealed JSON manifest plus a JSON/JSONL/CSV query-record file.
This evaluator is intentionally non-promotable: the Oracle receives only the
true old/new role, then predicts among every registered class in that role.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CLAIM_SCOPE = "LICENSED_ORACLE_UPPER_BOUND_NON_PROMOTABLE"
SCHEMA = "cvs.phase2.d92_role_oracle_125_manifest.v1"
SUMMARY_SCHEMA = "cvs.phase2.d92_role_oracle_125_summary.v1"
VARIANTS = ("baseline", "role_oracle")
STATES = ("before", "after")
ROLES = ("target_old", "target_new")
EXPECTED_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
EXPECTED_SEEDS = (713102, 713103, 713104, 713105, 713106)
EXPECTED_SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
EXPECTED_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FORMAL_QUERY_PER_TX = 20
METRICS = (
    "b_old_acc",
    "a_old_acc",
    "seen_new_acc",
    "h_old_new",
    "balanced_accuracy",
    "floor",
    "min_old",
    "min_new",
    "forgetting",
)
PAIR_BINDINGS = (
    "receiver",
    "seed",
    "k_shot",
    "new_class_count",
    "scenario",
    "state",
    "query_token",
    "true_class",
    "true_role",
    "score_contract_sha256",
    "model_state_sha256",
    "score_vector_sha256",
    "query_payload_sha256",
    "registered_classes_sha256",
    "old_registered_classes",
    "new_registered_classes",
)


class OracleSummaryError(RuntimeError):
    """Raised when the paired upper-bound evidence is not auditable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry_sha256(old_classes: Sequence[str], new_classes: Sequence[str]) -> str:
    payload = json.dumps(
        {"old": list(old_classes), "new": list(new_classes)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _classes(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise OracleSummaryError(f"{field} must be a JSON string array")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise OracleSummaryError(f"{field} contains duplicate classes")
    return result


def _load_records(path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "json":
        payload = _read_json(path)
        records = payload["records"] if isinstance(payload, dict) else payload
    elif fmt == "jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif fmt == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    else:
        raise OracleSummaryError(f"unsupported records_format: {fmt}")
    if not isinstance(records, list) or not all(isinstance(x, dict) for x in records):
        raise OracleSummaryError("records must be a list of objects")
    normalized: list[dict[str, Any]] = []
    required = {
        "row_id", "receiver", "seed", "k_shot", "new_class_count", "scenario",
        "state", "variant", "query_token", "true_class", "true_role",
        "predicted_class", "score_contract_sha256", "model_state_sha256",
        "score_vector_sha256", "query_payload_sha256", "registered_classes_sha256",
        "old_registered_classes", "new_registered_classes",
    }
    for index, raw in enumerate(records):
        missing = sorted(required - raw.keys())
        if missing:
            raise OracleSummaryError(f"record {index} missing fields: {missing}")
        row = dict(raw)
        row["seed"] = int(row["seed"])
        row["k_shot"] = int(row["k_shot"])
        row["new_class_count"] = int(row["new_class_count"])
        row["old_registered_classes"] = _classes(row["old_registered_classes"], "old_registered_classes")
        row["new_registered_classes"] = _classes(row["new_registered_classes"], "new_registered_classes")
        if row["variant"] not in VARIANTS or row["state"] not in STATES or row["true_role"] not in ROLES:
            raise OracleSummaryError(f"record {index} has invalid variant/state/role")
        for field in (
            "score_contract_sha256", "model_state_sha256", "score_vector_sha256",
            "query_payload_sha256", "registered_classes_sha256",
        ):
            value = str(row[field])
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise OracleSummaryError(f"record {index} {field} is not lowercase SHA256")
        normalized.append(row)
    return normalized


def _harmonic(old: float, new: float) -> float:
    return 0.0 if old + new == 0.0 else 2.0 * old * new / (old + new)


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return math.nan
    return sum(row["predicted_class"] == row["true_class"] for row in rows) / len(rows)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    before_old = [r for r in rows if r["state"] == "before" and r["true_role"] == "target_old"]
    after_old = [r for r in rows if r["state"] == "after" and r["true_role"] == "target_old"]
    after_new = [r for r in rows if r["state"] == "after" and r["true_role"] == "target_new"]
    if not before_old or not after_old or not after_new:
        raise OracleSummaryError("every aggregate must contain before-old, after-old, and after-new queries")
    b_old = _accuracy(before_old)
    a_old = _accuracy(after_old)
    seen_new = _accuracy(after_new)
    class_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in after_old + after_new:
        class_rows[str(row["true_class"])].append(row)
    per_class = {key: _accuracy(value) for key, value in class_rows.items()}
    old_classes = {str(r["true_class"]) for r in after_old}
    new_classes = {str(r["true_class"]) for r in after_new}
    return {
        "b_old_acc": b_old,
        "a_old_acc": a_old,
        "seen_new_acc": seen_new,
        "h_old_new": _harmonic(a_old, seen_new),
        "balanced_accuracy": sum(per_class.values()) / len(per_class),
        "floor": min(per_class.values()),
        "min_old": min(per_class[c] for c in old_classes),
        "min_new": min(per_class[c] for c in new_classes),
        "forgetting": b_old - a_old,
    }


def _paired_table(records: Sequence[dict[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[tuple(row[field] for field in group_fields)][row["variant"]].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda x: tuple(str(v) for v in x)):
        variants = grouped[key]
        if set(variants) != set(VARIANTS):
            raise OracleSummaryError("aggregate is missing a paired variant")
        base = _metrics(variants["baseline"])
        oracle = _metrics(variants["role_oracle"])
        item = {field: value for field, value in zip(group_fields, key)}
        for metric in METRICS:
            item[f"baseline_{metric}"] = base[metric]
            item[f"oracle_{metric}"] = oracle[metric]
            item[f"delta_{metric}"] = oracle[metric] - base[metric]
        output.append(item)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise OracleSummaryError(f"refusing to write empty table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_pairing(records: Sequence[dict[str, Any]], manifest: Mapping[str, Any]) -> None:
    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    row_meta: dict[str, tuple[Any, ...]] = {}
    row_scenarios: dict[str, set[str]] = defaultdict(set)
    class_coverage: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    class_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in records:
        rid = str(row["row_id"])
        meta = (row["receiver"], row["seed"], row["k_shot"], row["new_class_count"])
        if rid in row_meta and row_meta[rid] != meta:
            raise OracleSummaryError("row_id metadata drift")
        row_meta[rid] = meta
        row_scenarios[rid].add(str(row["scenario"]))
        if row["variant"] == "baseline":
            class_coverage[(rid, str(row["scenario"]), str(row["state"]))].add(str(row["true_class"]))
            class_counts[
                (rid, str(row["scenario"]), str(row["state"]), str(row["true_class"]))
            ] += 1
        key = (rid, str(row["state"]), str(row["scenario"]), str(row["query_token"]))
        if row["variant"] in pairs[key]:
            raise OracleSummaryError("duplicate variant for query pair")
        pairs[key][str(row["variant"])] = row
    if len(row_meta) != 125 or sum(len(v) for v in row_scenarios.values()) != 375:
        raise OracleSummaryError("fresh matrix must contain exactly 125 rows and 375 row-scenes")
    if len(set(row_meta.values())) != 125:
        raise OracleSummaryError("125 row_ids must map to 125 unique Cartesian row keys")
    expected_cartesian = {
        (receiver, int(seed), int(item["k_shot"]), int(item["new_class_count"]))
        for receiver in manifest["receivers"]
        for seed in manifest["seeds"]
        for item in manifest["slices"]
    }
    if len(expected_cartesian) != 125 or set(row_meta.values()) != expected_cartesian:
        raise OracleSummaryError("row metadata does not equal the sealed 125 Cartesian matrix")
    expected_scenarios = set(manifest["scenarios"])
    if len(expected_scenarios) != 3 or any(v != expected_scenarios for v in row_scenarios.values()):
        raise OracleSummaryError("every row must contain the same three sealed scenarios")
    for pair in pairs.values():
        if set(pair) != set(VARIANTS):
            raise OracleSummaryError("every query must have baseline and role_oracle records")
        baseline, oracle = pair["baseline"], pair["role_oracle"]
        for field in PAIR_BINDINGS:
            if baseline[field] != oracle[field]:
                raise OracleSummaryError(f"paired record mismatch: {field}")
        old_classes = set(oracle["old_registered_classes"])
        new_classes = set(oracle["new_registered_classes"])
        if old_classes & new_classes:
            raise OracleSummaryError("old/new registered class sets overlap")
        if len(old_classes) != int(manifest["old_class_count"]):
            raise OracleSummaryError("old registered class count drift")
        expected_registry_sha = _registry_sha256(
            oracle["old_registered_classes"], oracle["new_registered_classes"]
        )
        if oracle["registered_classes_sha256"] != expected_registry_sha:
            raise OracleSummaryError("registered class list/hash mismatch")
        all_registered = old_classes | new_classes
        if baseline["true_class"] not in all_registered or baseline["predicted_class"] not in all_registered:
            raise OracleSummaryError("baseline prediction is outside all registered classes")
        if oracle["state"] == "before":
            if oracle["true_role"] != "target_old" or new_classes:
                raise OracleSummaryError("before state must contain only old queries/classes")
            if oracle["predicted_class"] != baseline["predicted_class"]:
                raise OracleSummaryError("before predictions must be identical")
        else:
            if len(new_classes) != int(oracle["new_class_count"]):
                raise OracleSummaryError("after new registered class count drift")
            allowed = old_classes if oracle["true_role"] == "target_old" else new_classes
            if oracle["true_class"] not in allowed or oracle["predicted_class"] not in allowed:
                raise OracleSummaryError("role_oracle has a cross-role classification error")
            if baseline["predicted_class"] == baseline["true_class"] and oracle["predicted_class"] != oracle["true_class"]:
                raise OracleSummaryError("role_oracle violates per-query correctness monotonicity")
            baseline_allowed = (
                old_classes
                if oracle["true_role"] == "target_old"
                else new_classes
            )
            if (
                baseline["predicted_class"] in baseline_allowed
                and oracle["predicted_class"] != baseline["predicted_class"]
            ):
                raise OracleSummaryError(
                    "role_oracle changed an already within-role prediction"
                )
    for (rid, scenario, state), observed_classes in class_coverage.items():
        exemplar = next(
            row for row in records
            if row["row_id"] == rid and row["scenario"] == scenario and row["state"] == state
        )
        expected_classes = set(exemplar["old_registered_classes"])
        if state == "after":
            expected_classes.update(exemplar["new_registered_classes"])
        if observed_classes != expected_classes:
            raise OracleSummaryError("query truth does not cover every registered class")
        if any(
            class_counts[(rid, scenario, state, class_name)]
            != FORMAL_QUERY_PER_TX
            for class_name in expected_classes
        ):
            raise OracleSummaryError(
                "every row/scenario/state/class requires exactly 20 queries"
            )


def summarize(manifest_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    output_root = Path(output_root).resolve()
    manifest = _read_json(manifest_path)
    required_manifest = {
        "schema": SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "fresh_run": True,
        "job_count": 125,
        "scenario_pair_count": 375,
        "variants": list(VARIANTS),
        "fresh_no_oracle_same_run_paired": True,
        "historical_reference_d92_audit_complete": True,
        "historical_reference_d92_semantically_equivalent": True,
        "candidate": "d92_role_oracle_licensed_upper_bound",
        "formal_protocol_valid": False,
        "promotion_eligible": False,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise OracleSummaryError(f"manifest {key} must equal {expected!r}")
    if not manifest.get("fresh_pairing_id") or not manifest.get("run_id"):
        raise OracleSummaryError("fresh_pairing_id and run_id are required")
    if not all(key in manifest for key in ("receivers", "seeds", "slices", "old_class_count", "scenarios")):
        raise OracleSummaryError("manifest matrix axes are incomplete")
    manifest_slices = tuple(
        (int(item["k_shot"]), int(item["new_class_count"]))
        for item in manifest["slices"]
    )
    if (
        tuple(manifest["receivers"]) != EXPECTED_RECEIVERS
        or tuple(int(value) for value in manifest["seeds"]) != EXPECTED_SEEDS
        or manifest_slices != EXPECTED_SLICES
        or tuple(manifest["scenarios"]) != EXPECTED_SCENARIOS
    ):
        raise OracleSummaryError("manifest axes do not equal the frozen D92 125 matrix")
    records_path = (manifest_path.parent / manifest["records_path"]).resolve()
    if _sha256(records_path) != manifest.get("records_sha256"):
        raise OracleSummaryError("records SHA256 mismatch")
    records = _load_records(records_path, str(manifest["records_format"]))
    _validate_pairing(records, manifest)
    tables = {
        "row_metrics.csv": _paired_table(records, ("row_id", "receiver", "seed", "k_shot", "new_class_count")),
        "scene_metrics.csv": _paired_table(records, ("row_id", "receiver", "seed", "k_shot", "new_class_count", "scenario")),
        "receiver_slice_metrics.csv": _paired_table(records, ("receiver", "k_shot", "new_class_count")),
        "scene_slice_metrics.csv": _paired_table(records, ("scenario", "k_shot", "new_class_count")),
        "receiver_scene_slice_metrics.csv": _paired_table(records, ("receiver", "scenario", "k_shot", "new_class_count")),
        "seed_slice_metrics.csv": _paired_table(records, ("seed", "k_shot", "new_class_count")),
        "slice_metrics.csv": _paired_table(records, ("k_shot", "new_class_count")),
    }
    tolerance = 1e-12
    for name, rows in tables.items():
        for row in rows:
            for metric in METRICS:
                delta = float(row[f"delta_{metric}"])
                if metric == "forgetting":
                    if delta > tolerance:
                        raise OracleSummaryError(f"{name} forgetting monotonicity violated")
                elif delta < -tolerance:
                    raise OracleSummaryError(f"{name} {metric} monotonicity violated")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    for name, rows in tables.items():
        _write_csv(output_root / name, rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": CLAIM_SCOPE,
        "claim_scope": CLAIM_SCOPE,
        "promotable": False,
        "protocol_legal_performance": False,
        "oracle_information": "query_old_new_role_only_no_tx_identity",
        "decision_scope": "per_sample_all_registered_classes_within_provided_role",
        "fresh_pairing_id": manifest["fresh_pairing_id"],
        "run_id": manifest["run_id"],
        "candidate": manifest.get("candidate"),
        "job_count": 125,
        "scenario_pair_count": 375,
        "query_pair_count": len(records) // 2,
        "before_prediction_identity": "PASS",
        "same_score_state_token_pairing": "PASS",
        "oracle_cross_role_error_count": 0,
        "monotonicity": "PASS",
        "records_sha256": manifest["records_sha256"],
        "tables": {name: len(rows) for name, rows in tables.items()},
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.manifest, args.output_root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
