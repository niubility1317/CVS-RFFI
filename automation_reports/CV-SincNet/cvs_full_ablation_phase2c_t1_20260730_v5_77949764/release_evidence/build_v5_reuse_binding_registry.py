#!/usr/bin/env python3
"""Build the v5 registry by reusing only v4 COMPLETE prediction receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.full_ablation_spec import validate_plan_rows  # noqa: E402
from cvsrffi.stage2_ablation_factory import (  # noqa: E402
    resolved_stage2_config_hash,
)
from cvsrffi.stage2_ablation_release import (  # noqa: E402
    BINDING_REGISTRY_SCHEMA,
    RUNNER_SUMMARY_SCHEMA,
    SEALED_PLAN_SCHEMA,
    validate_binding_registry,
)


class V5ReuseRegistryError(ValueError):
    """Raised when v4 completion evidence cannot safely seed v5."""


EXPECTED_GIT_COMMIT = "779497647f1e616f1a143121635fdc183f3ec0bb"
PRIOR_RUN_ID = "cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise V5ReuseRegistryError(f"JSON object required: {path}")
    return payload


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing v5 registry")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_source_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        plan.get("schema") != "cvs.full_ablation.plan.v1"
        or plan.get("phase") != "phase2"
        or plan.get("phase2_matrix") != "stage2c"
        or plan.get("python_environment_id") != "CVS-RFFI"
        or plan.get("git_commit") != EXPECTED_GIT_COMMIT
    ):
        raise V5ReuseRegistryError("source plan is not the CVS-RFFI Stage2-C matrix")
    rows = list(plan.get("rows") or [])
    validate_plan_rows(rows)
    if len(rows) != 1425:
        raise V5ReuseRegistryError("v5 source plan must contain 1425 logical rows")
    return rows


def _require_prior_evidence(
    sealed: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if (
        sealed.get("schema") != SEALED_PLAN_SCHEMA
        or summary.get("schema") != RUNNER_SUMMARY_SCHEMA
        or summary.get("run_id") != sealed.get("run_id")
        or sealed.get("run_id") != PRIOR_RUN_ID
        or int(sealed.get("logical_row_count", -1)) != 1425
        or int(sealed.get("physical_execution_count", -1)) != 1350
        or int(summary.get("logical_row_count", -1)) != 1425
        or int(summary.get("physical_execution_count", -1)) != 1350
        or int(summary.get("completed_physical_count", -1)) != 641
        or int(summary.get("failed_physical_count", -1)) != 18
        or int(summary.get("not_launched_systemic_stop_count", -1)) != 691
        or summary.get("systemic_stop") is not True
        or summary.get("performance_values_visible_to_scheduler") is not False
    ):
        raise V5ReuseRegistryError("v4 sealed plan and terminal summary disagree")
    physical_rows = list(sealed.get("physical_rows") or [])
    statuses = list(summary.get("statuses") or [])
    by_status = {
        str(item.get("physical_execution_id")): dict(item)
        for item in statuses
    }
    by_physical = {
        str(item.get("physical_execution_id")): dict(item)
        for item in physical_rows
    }
    if (
        len(by_status) != 1350
        or len(by_physical) != 1350
        or set(by_status) != set(by_physical)
    ):
        raise V5ReuseRegistryError("v4 physical identity closure is incomplete")
    return physical_rows, by_status


def build_v5_registry(
    *,
    source_plan: Mapping[str, Any],
    base_registry: Mapping[str, Any],
    prior_sealed_plan: Mapping[str, Any],
    prior_runner_summary: Mapping[str, Any],
    receipt_sha256: Callable[[str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _require_source_plan(source_plan)
    row_keys = [str(row["row_key"]) for row in rows]
    prior_physical, statuses = _require_prior_evidence(
        prior_sealed_plan,
        prior_runner_summary,
    )
    base_by_key = validate_binding_registry(
        base_registry,
        expected_row_keys=row_keys,
    )
    if (
        base_registry.get("schema") != BINDING_REGISTRY_SCHEMA
        or base_registry.get("candidate_lock_sha256")
        != prior_sealed_plan.get("candidate_lock_sha256")
    ):
        raise V5ReuseRegistryError("base registry identity drift")
    source_by_key = {str(row["row_key"]): row for row in rows}

    prior_logical_sets: list[frozenset[str]] = []
    reused_physical_ids: list[str] = []
    reused_logical_count = 0
    output_by_key = {key: dict(value) for key, value in base_by_key.items()}
    prior_run_root = Path(str(prior_sealed_plan.get("run_root", "")))

    for physical in prior_physical:
        physical_id = str(physical["physical_execution_id"])
        logical_keys = frozenset(
            str(item["logical_row_key"])
            for item in physical.get("logical_rows") or []
        )
        if not logical_keys or not logical_keys.issubset(output_by_key):
            raise V5ReuseRegistryError(
                f"v4 physical row has invalid logical membership: {physical_id}"
            )
        prior_logical_sets.append(logical_keys)
        for logical in physical.get("logical_rows") or []:
            logical_key = str(logical["logical_row_key"])
            source_row = source_by_key[logical_key]
            if (
                str(logical.get("ablation_id"))
                != str(source_row["ablation_id"])
                or str(logical.get("effective_config_hash"))
                != resolved_stage2_config_hash(
                    str(source_row["ablation_id"])
                )
            ):
                raise V5ReuseRegistryError(
                    f"v5 effective configuration drift for {logical_key}"
                )
        status = statuses[physical_id]
        if status.get("status") != "COMPLETE":
            continue
        if (
            status.get("prediction_complete") is not True
            or status.get("scores_complete") is not True
            or int(status.get("predictor_return_code", -1)) != 0
            or any(
                int(value) != 0
                for value in status.get("scorer_return_codes") or []
            )
            or physical.get("mode") != "execute"
        ):
            raise V5ReuseRegistryError(
                f"v4 COMPLETE evidence is internally inconsistent: {physical_id}"
            )

        receipt_path = Path(str(physical["row_execution_receipt"]))
        expected_receipt = (
            prior_run_root
            / "physical"
            / physical_id
            / "row_execution_receipt.json"
        )
        if receipt_path != expected_receipt:
            raise V5ReuseRegistryError(
                f"v4 receipt path is outside the exact run row: {physical_id}"
            )
        receipt_hash = receipt_sha256(str(receipt_path))
        if len(receipt_hash) != 64 or any(
            char not in "0123456789abcdef" for char in receipt_hash
        ):
            raise V5ReuseRegistryError(
                f"invalid receipt SHA256 for {physical_id}"
            )

        representative_key = str(physical["representative_logical_row_key"])
        representative_binding = output_by_key[representative_key]
        for logical_key in logical_keys:
            binding = output_by_key[logical_key]
            for field in (
                "feature_cache_payload",
                "feature_cache_payload_sha256",
                "feature_cache_manifest",
                "feature_cache_manifest_sha256",
                "phase2_data_status",
                "capsule_id",
                "split_id",
                "phase1_bundle_sha256",
                "phase1_prototype_sha256",
                "scoring_manifest",
                "scoring_manifest_sha256",
            ):
                if binding[field] != representative_binding[field]:
                    raise V5ReuseRegistryError(
                        f"v4 alias binding drift for {physical_id}"
                    )
            binding["mode"] = "reuse_prediction"
            binding["reuse_row_execution_receipt"] = str(receipt_path)
            binding["reuse_row_execution_receipt_sha256"] = receipt_hash
            binding["reuse_physical_execution_id"] = physical_id
            reused_logical_count += 1
        reused_physical_ids.append(physical_id)

    if len(set().union(*prior_logical_sets)) != 1425:
        raise V5ReuseRegistryError("v4 physical groups do not cover all logical rows")
    if sum(len(group) for group in prior_logical_sets) != 1425:
        raise V5ReuseRegistryError("v4 physical groups overlap")

    bindings = [output_by_key[key] for key in row_keys]
    registry = {
        "schema": BINDING_REGISTRY_SCHEMA,
        "candidate_lock_sha256": str(
            base_registry["candidate_lock_sha256"]
        ),
        "bindings": bindings,
    }
    validate_binding_registry(registry, expected_row_keys=row_keys)
    reused_count = len(reused_physical_ids)
    execution_count = 1350 - reused_count
    if reused_count != 641 or execution_count != 709:
        raise V5ReuseRegistryError(
            f"unexpected v5 split: reuse={reused_count}, execute={execution_count}"
        )
    sorted_reused_ids = sorted(reused_physical_ids)
    audit = {
        "logical_row_count": 1425,
        "physical_execution_count": 1350,
        "reused_physical_count": reused_count,
        "execute_physical_count": execution_count,
        "reused_logical_count": reused_logical_count,
        "reused_physical_ids_sha256": hashlib.sha256(
            "\n".join(sorted_reused_ids).encode("utf-8")
        ).hexdigest(),
        "performance_values_read": False,
        "dataset_revalidated": False,
    }
    return registry, audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--base-binding-registry", required=True)
    parser.add_argument("--prior-sealed-plan", required=True)
    parser.add_argument("--prior-runner-summary", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Validate grouping/counts without reading receipts or writing a registry.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.structure_only:
        if args.output:
            raise V5ReuseRegistryError(
                "--structure-only cannot publish an output registry"
            )
        resolver = lambda _path: "0" * 64
    else:
        if not args.output:
            raise V5ReuseRegistryError("--output is required for publication")
        resolver = _sha256_file
    registry, audit = build_v5_registry(
        source_plan=_load_json(args.source_plan),
        base_registry=_load_json(args.base_binding_registry),
        prior_sealed_plan=_load_json(args.prior_sealed_plan),
        prior_runner_summary=_load_json(args.prior_runner_summary),
        receipt_sha256=resolver,
    )
    if not args.structure_only:
        _write_new(Path(args.output), registry)
        audit["output"] = str(Path(args.output))
        audit["output_sha256"] = _sha256_file(args.output)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
