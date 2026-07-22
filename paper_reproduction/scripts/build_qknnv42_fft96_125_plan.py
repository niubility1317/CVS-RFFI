#!/usr/bin/env python3
"""Build the matched-history 125-bundle qKNNV42+FFT96 Stage2-B/C plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
SEEDS = (713101, 713102, 713103, 713104, 713105)
K_VALUES = (1, 2, 5, 10, 20)
NEW_COUNTS = (5, 10, 20)
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return str(value).replace("-", "_")


def _remote_path(value: str, *, name: str) -> PurePosixPath:
    raw = str(value)
    if "\\" in raw:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(raw)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute POSIX path")
    return path


def _write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_source_packages(path: Path) -> tuple[list[str], dict[int, list[str]], dict[tuple[str, int], str]]:
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    if source.get("schema") != "cvs.phase2.adv3b02_ci_strict_plan.v1":
        raise ValueError("source strict plan schema drift")
    packages = source.get("packages")
    if not isinstance(packages, list):
        raise ValueError("source strict plan package list missing")
    old_labels: list[str] | None = None
    nested: dict[int, list[str]] = {}
    cache_by_receiver_seed: dict[tuple[str, int], str] = {}
    for item in packages:
        receiver = str(item["receiver"])
        seed = int(item["seed"])
        new_count = int(item["new_class_count"])
        if receiver not in RECEIVERS or seed not in SEEDS or new_count not in NEW_COUNTS:
            continue
        current_old = [str(value) for value in item["old_class_labels"]]
        current_new = [str(value) for value in item["new_class_labels"]]
        if old_labels is None:
            old_labels = current_old
        elif current_old != old_labels:
            raise ValueError("source old-class order drift")
        if new_count in nested and nested[new_count] != current_new:
            raise ValueError("source nested new-class order drift")
        nested[new_count] = current_new
        cache = str(item["target_cache_set"])
        key = (receiver, seed)
        if key in cache_by_receiver_seed and cache_by_receiver_seed[key] != cache:
            raise ValueError("source target-cache path drift")
        cache_by_receiver_seed[key] = cache
    if old_labels is None or set(nested) != set(NEW_COUNTS):
        raise ValueError("source class split is incomplete")
    if nested[20][:10] != nested[10] or nested[10][:5] != nested[5]:
        raise ValueError("source new-class sets are not nested")
    expected_keys = {(receiver, seed) for receiver in RECEIVERS for seed in SEEDS}
    if set(cache_by_receiver_seed) != expected_keys:
        raise ValueError("source target-cache coverage is incomplete")
    return old_labels, nested, cache_by_receiver_seed


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_plan = Path(args.source_strict_plan).resolve(strict=True)
    old_labels, nested, cache_by_receiver_seed = _load_source_packages(source_plan)
    local_artifact_paths = {
        "base_runtime": Path(args.base_runtime).resolve(strict=True),
        "candidate_lock": Path(args.candidate_lock).resolve(strict=True),
        "adapter": Path(args.adapter).resolve(strict=True),
        "head_artifact": Path(args.head_artifact).resolve(strict=True),
        "tta_policy": Path(args.tta_policy).resolve(strict=True),
    }
    remote_artifact_paths = {
        "base_runtime": _remote_path(
            args.remote_base_runtime, name="remote base runtime"
        ),
        "candidate_lock": _remote_path(
            args.remote_candidate_lock, name="remote candidate lock"
        ),
        "adapter": _remote_path(args.remote_adapter, name="remote adapter"),
        "head_artifact": _remote_path(
            args.remote_head_artifact, name="remote head artifact"
        ),
        "tta_policy": _remote_path(
            args.remote_tta_policy, name="remote TTA policy"
        ),
    }
    artifacts = {
        key: {
            "path": str(remote_artifact_paths[key]),
            "sha256": _sha256(local_path),
        }
        for key, local_path in local_artifact_paths.items()
    }
    if artifacts["base_runtime"]["sha256"] != (
        "b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9"
    ):
        raise ValueError("ADV3B02 identity TorchScript runtime SHA drift")
    run_root = _remote_path(args.run_root, name="run root")
    remote_source_plan = _remote_path(
        args.remote_source_strict_plan, name="remote source strict plan"
    )
    packages: list[dict[str, Any]] = []
    package_ids: dict[tuple[str, int, int], str] = {}
    for receiver in RECEIVERS:
        for seed in SEEDS:
            cache_set = cache_by_receiver_seed[(receiver, seed)]
            states = [(0, [])] + [(count, nested[count]) for count in NEW_COUNTS]
            for new_count, new_labels in states:
                suffix = "before" if new_count == 0 else f"after_new_{new_count}"
                package_id = f"rx_{_safe(receiver)}__seed_{seed}__{suffix}"
                package_parent = run_root / "packages" / package_id
                packages.append(
                    {
                        "package_id": package_id,
                        "registration_state": (
                            "before_registration"
                            if new_count == 0
                            else "after_registration"
                        ),
                        "stage": "stage2b" if new_count == 0 else "stage2c",
                        "receiver": receiver,
                        "seed": seed,
                        "new_class_count": new_count,
                        "old_class_labels": old_labels,
                        "new_class_labels": list(new_labels),
                        "reference_new_class_labels": (
                            list(nested[20]) if new_count == 0 else []
                        ),
                        "target_cache_set": cache_set,
                        "predictor_package_root": str(package_parent / "predictor"),
                        "scorer_root": str(package_parent / "scorer"),
                        "detached_seal": str(package_parent / "predictor.seal.json"),
                        "build_receipt": str(package_parent / "package_build_receipt.json"),
                        "pre_run_evidence_root": str(package_parent / "pre_run_evidence"),
                    }
                )
                package_ids[(receiver, seed, new_count)] = package_id
    bundles: list[dict[str, Any]] = []
    state_cells: list[dict[str, Any]] = []
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for k_shot in K_VALUES:
                bundle_id = f"rx_{_safe(receiver)}__seed_{seed}__k_{k_shot}"
                states = [
                    ("before_registration", 0),
                    ("after_registration", 5),
                    ("after_registration", 10),
                    ("after_registration", 20),
                ]
                cell_ids = []
                for registration_state, new_count in states:
                    suffix = "before" if new_count == 0 else f"after_new_{new_count}"
                    cell_id = f"{bundle_id}__{suffix}"
                    cell_ids.append(cell_id)
                    state_cells.append(
                        {
                            "cell_id": cell_id,
                            "bundle_id": bundle_id,
                            "package_id": package_ids[(receiver, seed, new_count)],
                            "registration_state": registration_state,
                            "receiver": receiver,
                            "seed": seed,
                            "seed_role": (
                                "development"
                                if seed == 713101
                                else "matched_history_confirmation"
                            ),
                            "k_shot": k_shot,
                            "new_class_count": new_count,
                            "output_root": str(run_root / "cells" / cell_id),
                        }
                    )
                bundles.append(
                    {
                        "bundle_id": bundle_id,
                        "receiver": receiver,
                        "seed": seed,
                        "seed_role": (
                            "development"
                            if seed == 713101
                            else "matched_history_confirmation"
                        ),
                        "k_shot": k_shot,
                        "state_cell_ids": cell_ids,
                    }
                )
    plan = {
        "schema": "cvs.phase2.qknnv42_fft96_125_plan.v1",
        "experiment_id": str(args.experiment_id),
        "run_root": str(run_root),
        "source_strict_plan": str(remote_source_plan),
        "source_strict_plan_sha256": _sha256(source_plan),
        "method": "QKNNV42_FFT96_SUPPORT_ONLY_V1",
        "base_model": "ADV3B02_CORE90_SOFT_E200",
        "base_checkpoint_lineage_sha256": (
            "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
        ),
        "receivers": list(RECEIVERS),
        "seeds": list(SEEDS),
        "development_seed": 713101,
        "k_values": list(K_VALUES),
        "official_forgetting_anchor_k_values": [1, 5, 10, 20],
        "new_class_counts": list(NEW_COUNTS),
        "scenarios": list(SCENARIOS),
        "old_class_labels": old_labels,
        "nested_new_class_labels": {str(key): nested[key] for key in NEW_COUNTS},
        "support_pool_max_k": 20,
        "query_per_tx": 20,
        "view_policy": "one_preoverlaid_leo_weak_view_per_scenario",
        "bundle_definition": (
            "5 receivers x 5 matched-history seeds x 5 K values; every bundle "
            "contains one physical Stage2-B package with an old-only registry/support "
            "and unregistered Y_new^20 reference query, plus three physical Stage2-C "
            "packages for nested new5/new10/new20"
        ),
        "independent_confirmation_claim_allowed": False,
        "result_boundary": (
            "matched-history 125-bundle diagnostic; seed 713101 is development"
        ),
        "artifacts": artifacts,
        "packages": packages,
        "bundles": bundles,
        "state_cells": state_cells,
        "counts": {
            "packages": len(packages),
            "bundles": len(bundles),
            "state_cells": len(state_cells),
            "scenario_rows": len(state_cells) * len(SCENARIOS),
        },
        "smoke_bundle_id": "rx_20_1__seed_713101__k_10",
        "launch_authority": False,
        "authority_state": "REAL_N607_BWRAP_STRACE_SMOKE_REQUIRED",
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
        "phase2_source_sample_access": False,
        "phase2_source_cache_access": False,
        "phase2_source_label_access": False,
        "phase2_source_derived_signal_access": False,
        "phase2_source_replay": False,
        "phase2_external_source_adapter_access": False,
    }
    if plan["counts"] != {
        "packages": 100,
        "bundles": 125,
        "state_cells": 500,
        "scenario_rows": 1500,
    }:
        raise ValueError("qKNNV42+FFT96 125 matrix count drift")
    _write_new(Path(args.output), plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--source-strict-plan", type=Path, required=True)
    parser.add_argument("--remote-source-strict-plan", required=True)
    parser.add_argument("--base-runtime", type=Path, required=True)
    parser.add_argument("--remote-base-runtime", required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--remote-candidate-lock", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--remote-adapter", required=True)
    parser.add_argument("--head-artifact", type=Path, required=True)
    parser.add_argument("--remote-head-artifact", required=True)
    parser.add_argument("--tta-policy", type=Path, required=True)
    parser.add_argument("--remote-tta-policy", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
