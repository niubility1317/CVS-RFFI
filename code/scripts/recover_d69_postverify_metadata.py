#!/usr/bin/env python3
"""Recover D69 metadata after the v0 lifecycle-count-only verifier failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--executed-probe", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    probe_path = args.executed_probe.resolve()
    metadata_path = output / "D69_PROBE_METADATA.json"
    if metadata_path.exists():
        raise RuntimeError("D69 metadata already exists; recovery refuses overwrite")
    for name in (
        "RECEIPT.json",
        "selection.json",
        "support_audit.json",
        "geometry_audit.json",
        "resource_audit.json",
        "training_log.jsonl",
    ):
        if not (output / name).is_file():
            raise RuntimeError(f"missing sealed runner artifact: {name}")

    probe = _load("d69_executed_probe_recovery", probe_path)
    support = json.loads((output / "support_audit.json").read_text(encoding="utf-8"))
    source_closure = support["candidate_lock"]["source_closure"]
    helper_hashes = {
        key: value for key, value in source_closure.items() if key.startswith("d69_")
    }
    probe_sha = _sha256(probe_path)
    evidence = probe._verify_output(output, probe_sha, helper_hashes)

    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    target = [
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    records: list[dict[str, Any]] = []
    for row in target:
        for phase, field in (
            ("before", "before_covariance_audit"),
            ("final", "final_covariance_audit"),
        ):
            audit = row["geometry_summary"][field]
            records.append(
                {
                    "candidate_id": row["candidate_id"],
                    "scenario": row["scenario"],
                    "fold_index": int(row["fold_index"]),
                    "phase": phase,
                    "d69_phase": audit["d69_phase"],
                    "before_old_row_sha256": audit["d69_before_old_row_sha256"],
                    "joint_d62_row_sha256": audit["d69_joint_d62_row_sha256"],
                    "actual_row_sha256": audit["d69_actual_row_sha256"],
                    "old_rows_unchanged": audit[
                        "d69_old_row_fp32_bitwise_unchanged"
                    ],
                    "new_rows_match_joint": audit[
                        "d69_new_row_fp32_matches_joint_d62"
                    ],
                }
            )
    if len(records) != 60:
        raise RuntimeError(f"expected 60 fit audits, got {len(records)}")
    before_count = sum(record["phase"] == "before" for record in records)
    final_count = sum(record["phase"] == "final" for record in records)
    if before_count != 30 or final_count != 30:
        raise RuntimeError("D69 before/final recovery pair count drift")
    if not all(
        record["old_rows_unchanged"] and record["new_rows_match_joint"]
        for record in records
    ):
        raise RuntimeError("D69 recovered lifecycle row identity drift")
    record_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt = json.loads((output / "RECEIPT.json").read_text(encoding="utf-8"))
    metadata = {
        "schema": "cvs.phase2.d69.frozen_d62_old_append_d62_new_probe.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_PROBE",
        "arm": probe.ARM,
        "formal_candidate": False,
        "probe_forced_nonpromotable": True,
        "selected_only_full_k10_refit_allowed": False,
        "query_opened": False,
        "probe_script_sha256": probe_sha,
        "formula": probe.FORMULA,
        "lifecycle_fit_record_count": 60,
        "lifecycle_completed_pair_count": 30,
        "lifecycle_record_sha256": record_sha,
        "component_fit_execution_count": 1080,
        "recovery": {
            "reason": "v0 post-run verifier expected 15 pairs instead of 30 precision-specific pairs",
            "runner_rerun": False,
            "sealed_runner_artifacts_modified": False,
            "executed_probe_path": str(probe_path),
            "recovery_script_sha256": _sha256(Path(__file__).resolve()),
            "receipt_sha256_before_recovery": _sha256(output / "RECEIPT.json"),
            "training_log_sha256_before_recovery": _sha256(
                output / "training_log.jsonl"
            ),
        },
        "runtime_root": receipt.get("runtime_root"),
        "probe_root": str(probe_path.parents[2]),
        **evidence,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
