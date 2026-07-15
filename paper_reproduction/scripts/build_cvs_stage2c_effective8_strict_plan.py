#!/usr/bin/env python3
"""Build the fail-closed Landlock/memfd effective8 Stage2-C 300-cell plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.scripts.build_cvs_stage2c_effective8_formal_plan import (
    K_VALUES,
    NEW_COUNTS,
    RECEIVERS,
    SCENARIOS,
    SEEDS,
    generate_plan as generate_base_plan,
)


SCHEMA = "cvs.stage2c.effective8.landlock_strict_matrix.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _runtime_path(root: str, relative: str) -> str:
    return str(PurePosixPath(root) / PurePosixPath(relative))


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_receiver(receiver: str) -> str:
    return receiver.replace("-", "_")


def validate_strict_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA:
        raise ValueError("strict matrix schema drift")
    exact = {
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "smoke_authority": True,
    }
    if any(plan.get(key) != value for key, value in exact.items()):
        raise ValueError("strict matrix Phase2 contract drift")
    if plan.get("launch_authority") not in (False, True):
        raise ValueError("strict matrix launch authority must be boolean")
    capsule_sha = str(plan.get("candidate_capsule_sha256", ""))
    if SHA256_RE.fullmatch(capsule_sha) is None:
        raise ValueError("strict matrix candidate capsule trust root is invalid")
    counts = plan.get("expected_counts")
    expected = {
        "target_cache_sets": 25,
        "sealed_predictor_packages": 75,
        "prediction_cells": 300,
        "formal_scenario_rows": 900,
    }
    if counts != expected:
        raise ValueError("strict matrix declared counts drift")
    caches = list(plan.get("cache_steps", []))
    packages = list(plan.get("package_steps", []))
    if len(caches) != 25 or len(packages) != 75:
        raise ValueError("strict matrix cache/package count drift")
    cache_ids = {str(item.get("cache_id")) for item in caches}
    identities: set[tuple[str, int, int]] = set()
    cell_ids: set[str] = set()
    for raw in packages:
        item = dict(raw)
        identity = (
            str(item.get("receiver")),
            int(item.get("seed", -1)),
            int(item.get("new_class_count", -1)),
        )
        if identity in identities:
            raise ValueError("strict matrix package identity duplication")
        identities.add(identity)
        if str(item.get("cache_id")) not in cache_ids:
            raise ValueError("strict matrix package/cache binding drift")
        cells = list(item.get("cells", []))
        if tuple(int(cell.get("k_shot", -1)) for cell in cells) != K_VALUES:
            raise ValueError("strict matrix K-shot coverage drift")
        for cell in cells:
            cell_id = str(cell.get("cell_id", ""))
            if not cell_id or cell_id in cell_ids:
                raise ValueError("strict matrix cell identity drift")
            cell_ids.add(cell_id)
    expected_identities = {
        (receiver, seed, new_count)
        for receiver in RECEIVERS
        for seed in SEEDS
        for new_count in NEW_COUNTS
    }
    if identities != expected_identities or len(cell_ids) != 300:
        raise ValueError("strict matrix exact cell coverage drift")
    return dict(plan)


def generate_strict_plan(
    plan_path: Path,
    *,
    out_dir: Path,
    runtime_project_root: str,
    runtime_artifact_root: str,
    expected_candidate_capsule_sha256: str,
    strict_run_suffix: str = "landlock_strict300",
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite strict plan directory: {out_dir}")
    if SHA256_RE.fullmatch(expected_candidate_capsule_sha256.lower()) is None:
        raise ValueError("candidate capsule SHA256 must be an external 64-hex trust root")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", strict_run_suffix) is None:
        raise ValueError("strict run suffix must be one safe path component")
    base_dir = out_dir / "base_plan"
    base = generate_base_plan(
        plan_path,
        out_dir=base_dir,
        runtime_project_root=runtime_project_root,
    )
    source = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    split_path = REPO_ROOT / source["class_split_manifest"]
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    old_labels = [str(value) for value in split["target_old_tx_labels"]]
    new_by_count = {
        int(key): [str(value) for value in values]
        for key, values in split["nested_target_new_tx_labels"].items()
    }
    experiment_id = f"{source['experiment_id']}_{strict_run_suffix}"
    run_root = _runtime_path(runtime_project_root, f"runs/{experiment_id}")
    source_run_root = _runtime_path(
        runtime_project_root, f"runs/{source['experiment_id']}"
    )
    artifact = {
        "candidate_lock": _runtime_path(source_run_root, "candidate_lock_v2.json"),
        "base_runtime": _runtime_path(runtime_artifact_root, "base_runtime.ts"),
        "candidate_runtime": _runtime_path(runtime_artifact_root, "candidate_runtime.ts"),
        "candidate_capsule": _runtime_path(runtime_artifact_root, "candidate_capsule.json"),
        "runtime_config_receipt": _runtime_path(runtime_artifact_root, "runtime_configs/runtime_config_receipt.json"),
        "runtime_adapter": _runtime_path(runtime_artifact_root, "runtime_configs/effective8_runtime_adapter.json"),
        "runtime_head": _runtime_path(runtime_artifact_root, "runtime_configs/effective8_runtime_head.json"),
        "runtime_tta": _runtime_path(runtime_artifact_root, "runtime_configs/effective8_runtime_tta.json"),
        "runtime_closure_root": _runtime_path(runtime_artifact_root, "runtime_closure"),
    }
    cache_steps: list[dict[str, Any]] = []
    cache_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    command_by_identity: dict[tuple[str, int], list[str]] = {}
    for contract, command in zip(
        base["target_cache_contracts"], base["commands"]["target_cache_build"]
    ):
        identity = (str(contract["receiver"]), int(contract["seed"]))
        command_by_identity[identity] = list(command)
        cache_by_identity[identity] = dict(contract)
    for receiver in RECEIVERS:
        for seed in SEEDS:
            contract = cache_by_identity[(receiver, seed)]
            cache_id = f"rx_{_safe_receiver(receiver)}__seed_{seed}"
            cache_steps.append(
                {
                    "cache_id": cache_id,
                    "receiver": receiver,
                    "seed": seed,
                    "cache_set_manifest": contract["cache_set_manifest"],
                    "build_command": command_by_identity[(receiver, seed)],
                }
            )
    package_steps: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            cache_id = f"rx_{_safe_receiver(receiver)}__seed_{seed}"
            cache_manifest = cache_by_identity[(receiver, seed)]["cache_set_manifest"]
            for new_count in NEW_COUNTS:
                package_id = f"{cache_id}__new_{new_count}"
                package_root = _runtime_path(run_root, f"packages/{package_id}/predictor")
                scorer_root = _runtime_path(run_root, f"packages/{package_id}/scorer")
                seal_path = _runtime_path(run_root, f"packages/{package_id}/predictor.seal.json")
                cells = []
                for k_shot in K_VALUES:
                    cell_id = f"{package_id}__k_{k_shot}"
                    cell_root = _runtime_path(run_root, f"cells/{cell_id}")
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "k_shot": k_shot,
                            "request_json": _runtime_path(cell_root, "request.json"),
                            "pre_run_evidence_root": _runtime_path(cell_root, "pre_run_evidence"),
                            "predictor_output_root": _runtime_path(cell_root, "predictor_output"),
                            "scoring_output_root": _runtime_path(cell_root, "scoring_output"),
                            "cell_receipt": _runtime_path(cell_root, "cell_receipt.json"),
                        }
                    )
                package_steps.append(
                    {
                        "package_id": package_id,
                        "cache_id": cache_id,
                        "receiver": receiver,
                        "seed": seed,
                        "new_class_count": new_count,
                        "target_cache_set": cache_manifest,
                        "old_class_labels": old_labels,
                        "new_class_labels": new_by_count[new_count],
                        "predictor_package_root": package_root,
                        "scorer_root": scorer_root,
                        "detached_seal": seal_path,
                        "cells": cells,
                    }
                )
    manifest = {
        "schema": SCHEMA,
        "experiment_id": experiment_id,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "phase2_clean_dataset_reachable": False,
        "phase2_clean_cache_reachable": False,
        "phase2_clean_control_flow_reachable": False,
        "phase2_pretrained_artifact_policy": "sealed_phase1_checkpoint_only",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "candidate_capsule_sha256": expected_candidate_capsule_sha256.lower(),
        "runtime_project_root": runtime_project_root,
        "runtime_artifact_root": runtime_artifact_root,
        "runtime_artifacts": artifact,
        "target_dataset_forbidden_root": _runtime_path(
            runtime_project_root, str(source["datasets"]["target"])
        ),
        "python_executable": "python",
        "strace_executable": "/usr/bin/strace",
        "landlock_launcher": _runtime_path(
            runtime_project_root, "code/scripts/run_phase2_landlock_isolated.py"
        ),
        "support_pool_max_k": 20,
        "query_per_tx": 20,
        "scenarios": list(SCENARIOS),
        "smoke_package_id": "rx_20_1__seed_713101__new_20",
        "smoke_k_shot": 1,
        "smoke_authority": True,
        "launch_authority": False,
        "authority_state": "N607_LANDLOCK_SMOKE_REQUIRED",
        "expected_counts": {
            "target_cache_sets": 25,
            "sealed_predictor_packages": 75,
            "prediction_cells": 300,
            "formal_scenario_rows": 900,
        },
        "base_plan_manifest": _runtime_path(
            source_run_root, "protocol_plan/plan_manifest.json"
        ),
        "cache_steps": cache_steps,
        "package_steps": package_steps,
    }
    validate_strict_plan(manifest)
    if manifest["launch_authority"] is not False:
        raise ValueError("generated strict plan must remain fail-closed before N607 smoke")
    # generate_base_plan created base_dir and its parent out_dir.
    _write_new(out_dir / "strict_plan_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runtime-project-root", required=True)
    parser.add_argument("--runtime-artifact-root", required=True)
    parser.add_argument("--expected-candidate-capsule-sha256", required=True)
    parser.add_argument("--strict-run-suffix", default="landlock_strict300")
    args = parser.parse_args(argv)
    result = generate_strict_plan(
        args.plan,
        out_dir=args.out_dir,
        runtime_project_root=args.runtime_project_root,
        runtime_artifact_root=args.runtime_artifact_root,
        expected_candidate_capsule_sha256=args.expected_candidate_capsule_sha256,
        strict_run_suffix=args.strict_run_suffix,
    )
    print(json.dumps(result["expected_counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
