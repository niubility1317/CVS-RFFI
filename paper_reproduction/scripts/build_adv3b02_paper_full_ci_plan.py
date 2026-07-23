#!/usr/bin/env python3
"""Build the 100-package/800-cell paper-mechanism ADV3B02 CI plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METHODS = ("csil_paper_full", "mopc_hr_paper_full")
OFFICIAL_METHODS = ("csil_official_repo", "mopc_hr_official_repo")
NEW_COUNTS = (2, 5, 10, 20)
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _safe(value: str) -> str:
    return str(value).replace("-", "_")


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build(args: argparse.Namespace) -> dict:
    methods = tuple(
        value.strip()
        for value in str(getattr(args, "methods", "")).split(",")
        if value.strip()
    ) or METHODS
    if len(set(methods)) != len(methods):
        raise ValueError("methods must not contain duplicates")
    official_execution = bool(methods) and all(
        value in OFFICIAL_METHODS for value in methods
    )
    if methods != METHODS and not official_execution:
        raise ValueError(
            "methods must be the legacy pair or a nonempty official-repository subset"
        )
    requested_counts = str(getattr(args, "new_counts", "")).strip()
    new_counts = (
        tuple(int(value) for value in requested_counts.split(",") if value)
        if requested_counts
        else NEW_COUNTS
    )
    if (
        not new_counts
        or tuple(sorted(set(new_counts))) != new_counts
        or any(value <= 0 for value in new_counts)
    ):
        raise ValueError(
            "new-counts must be unique positive values in ascending order"
        )
    split_path = Path(args.class_split).resolve(strict=True)
    split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    old = [str(value) for value in split["target_old_tx_labels"]]
    receivers = [str(value) for value in split["target_receiver_labels"]]
    seeds = [int(value) for value in split["confirmation_seeds"]]
    k_values = [int(value) for value in split["k_values"]]
    if receivers != ["20-1", "3-19", "7-14", "7-7", "8-8"]:
        raise ValueError("target receiver matrix drift")
    if seeds != [713101, 713102, 713103, 713104, 713105]:
        raise ValueError("confirmation seed matrix drift")
    if k_values != [1, 5, 10, 20]:
        raise ValueError("K matrix drift")
    if len(old) != 6 or len(set(old)) != 6:
        raise ValueError("official comparison requires exactly six unique old classes")
    nested = {
        int(key): [str(item) for item in values]
        for key, values in split["nested_target_new_tx_labels"].items()
    }
    if tuple(sorted(nested)) != new_counts:
        raise ValueError("new-class count matrix drift")
    previous = []
    for new_count in new_counts:
        current = nested[new_count]
        if (
            len(current) != new_count
            or len(set(current)) != new_count
            or current[: len(previous)] != previous
        ):
            raise ValueError("new-class sets are not nested exact-size prefixes")
        if set(old) & set(current):
            raise ValueError("old/new class sets overlap")
        previous = current
    required_total_capacity = int(
        getattr(args, "required_total_capacity", 0) or (len(old) + new_counts[-1])
    )
    if required_total_capacity < len(old) + new_counts[-1]:
        raise ValueError("required total capacity is smaller than the class registry")
    expected_cache_scope = str(
        getattr(args, "expected_cache_scope", "stage2_registered")
    )
    if expected_cache_scope not in {
        "stage2_registered",
        "external_comparison_registered",
    }:
        raise ValueError("unsupported comparison cache scope")
    cache_parity_root_raw = getattr(args, "cache_parity_root", None)
    cache_parity_root = (
        Path(cache_parity_root_raw) if cache_parity_root_raw else None
    )
    if (
        expected_cache_scope == "external_comparison_registered"
        and cache_parity_root is None
    ):
        raise ValueError("external comparison plans require cache parity receipts")
    parity_reference_root_raw = getattr(
        args, "parity_reference_cache_root", None
    )
    parity_reference_root = (
        Path(parity_reference_root_raw) if parity_reference_root_raw else None
    )
    parity_preserved_labels = tuple(
        value.strip()
        for value in str(
            getattr(args, "parity_preserved_class_labels", "")
        ).split(",")
        if value.strip()
    )
    if expected_cache_scope == "external_comparison_registered":
        parity_reference_new20 = tuple(
            str(value)
            for value in split.get("parity_reference_new20_tx_labels", [])
        )
        if parity_reference_root is None:
            raise ValueError(
                "external comparison plans require a parity reference cache root"
            )
        if (
            len(parity_preserved_labels) != 20
            or len(set(parity_preserved_labels)) != 20
            or len(parity_reference_new20) != 20
            or len(set(parity_reference_new20)) != 20
            or tuple(parity_preserved_labels) != parity_reference_new20
        ):
            raise ValueError(
                "external comparison parity requires exact first20 new labels"
            )
    artifact_paths = {
        "base_checkpoint": Path(args.base_checkpoint).resolve(strict=True),
        "candidate_lock": Path(args.candidate_lock).resolve(strict=True),
        "adapter": Path(args.adapter).resolve(strict=True),
        "head_artifact": Path(args.head_artifact).resolve(strict=True),
        "tta_policy": Path(args.tta_policy).resolve(strict=True),
    }
    artifacts = {
        key: {"path": str(path), "sha256": _sha256(path)}
        for key, path in artifact_paths.items()
    }
    run_root = Path(args.run_root)
    cache_root = Path(args.target_cache_root)
    packages = []
    cells = []
    for receiver in receivers:
        for seed in seeds:
            cache_set = (
                cache_root / f"rx_{_safe(receiver)}" / f"seed_{seed}" / "cache_set.json"
            )
            for new_count in new_counts:
                package_id = f"rx_{_safe(receiver)}__seed_{seed}__new_{new_count}"
                package_parent = run_root / "packages" / package_id
                packages.append(
                    {
                        "package_id": package_id,
                        "receiver": receiver,
                        "seed": seed,
                        "new_class_count": new_count,
                        "old_class_labels": old,
                        "new_class_labels": nested[new_count],
                        "target_cache_set": str(cache_set),
                        "cache_parity_receipt": (
                            str(
                                cache_parity_root
                                / f"rx_{_safe(receiver)}"
                                / f"seed_{seed}.json"
                            )
                            if cache_parity_root is not None
                            else None
                        ),
                        "cache_parity_reference_cache_set": (
                            str(
                                parity_reference_root
                                / f"rx_{_safe(receiver)}"
                                / f"seed_{seed}"
                                / "cache_set.json"
                            )
                            if parity_reference_root is not None
                            else None
                        ),
                        "predictor_package_root": str(package_parent / "predictor"),
                        "scorer_root": str(package_parent / "scorer"),
                        "detached_seal": str(package_parent / "predictor.seal.json"),
                        "build_receipt": str(package_parent / "package_build_receipt.json"),
                    }
                )
                for method in methods:
                    for k_shot in k_values:
                        cell_id = f"{package_id}__{method}__k_{k_shot}"
                        cells.append(
                            {
                                "cell_id": cell_id,
                                "package_id": package_id,
                                "receiver": receiver,
                                "seed": seed,
                                "new_class_count": new_count,
                                "method": method,
                                "k_shot": k_shot,
                                "output_root": str(run_root / "cells" / cell_id),
                            }
                        )
    smoke_cell_ids = list(
        dict.fromkeys(
            [
                f"rx_20_1__seed_713101__new_{new_counts[0]}__{method}__k_1"
                for method in methods
            ]
            + [
                f"rx_20_1__seed_713101__new_{new_counts[-1]}__{method}__k_20"
                for method in methods
            ]
        )
    )
    plan = {
        "schema": "cvs.phase2.adv3b02_paper_full_ci_plan.v1",
        "experiment_id": str(args.experiment_id),
        "run_root": str(run_root),
        "predictor_script": (
            "paper_reproduction/scripts/"
            "run_adv3b02_paper_full_ci_truth_free_predictor.py"
        ),
        "claim_boundary": "formal_paper_method_comparison_baseline",
        "comparison_method_protocol_scope": (
            "stage2_main_method_protocol_exempt_new_class_leo_required"
        ),
        "new_class_support_channel_policy": "leo_satellite_required",
        "new_class_query_channel_policy": "leo_satellite_required",
        "base_source_reference_access_allowed": True,
        "fixed_received_iq_reused_across_epochs": True,
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
        "query_labels_used_for_training": False,
        "query_evaluated_after_training": True,
        "backbone": "ADV3B02",
        "backbone_uniformly_frozen": False,
        "methods": list(methods),
        "execution_semantics": (
            "OFFICIAL_CODE_EXECUTION_SEMANTICS"
            if official_execution
            else "LEGACY_PAPER_CODE_HYBRID"
        ),
        "receivers": receivers,
        "seeds": seeds,
        "k_values": k_values,
        "new_class_counts": list(new_counts),
        "required_total_capacity": required_total_capacity,
        "expected_cache_scope": expected_cache_scope,
        "parity_preserved_class_labels": list(parity_preserved_labels),
        "scenarios": list(SCENARIOS),
        "support_pool_max_k": 20,
        "query_per_tx": 20,
        "paper_parameter_lock": {
            "csil": {
                "epochs": 3,
                "batch_size": 20,
                "learning_rate": "0.01/(1+0.01*iteration)",
                "momentum": 0.9,
                "l2_factor": 0.05,
                "kd_weight": 0.2,
                "ewc_weight": 1.0,
            },
            "mopc_hr": {
                "epochs": 20,
                "batch_size": 16,
                "optimizer": "SGD",
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0002,
                "prototype_noise_std": 0.05,
                "prototype_momentum_alpha": 0.97,
                "beta": 1.0,
                "lambda_max": 1.0,
            },
        },
        "official_code_execution_lock": (
            {
                "csil": {
                    "entry": "ContinualLearning/WorkStage/CSIL.m",
                    "base_epochs": 20,
                    "base_batch_size": 128,
                    "increment_epochs": 3,
                    "increment_batch_size": 20,
                    "backbone_frozen_during_increment": True,
                    "new_dimension": "new_class_count",
                    "fisher_objective": "mean_log_shifted_softmax_then_exp_grad_squared",
                    "kd": "sum_squared_divide_32_weight_0.2",
                    "ewc": "sum_fisher_squared_delta_divide_2",
                },
                "mopc_hr": {
                    "entry": "MoPC_HR_trainer.py",
                    "base_epochs": 20,
                    "increment_epochs": 20,
                    "batch_size": 16,
                    "hr": "per_parameter_unsquared_l2",
                    "hr_effective_coefficient": 1.0,
                    "prototype_similarity": "raw_dot_then_softmax",
                    "prototype_logit_temperature": 2.0,
                    "kd_in_total_loss": False,
                    "query_decision": "all_registered_classifier_logits",
                },
                "base_sample_count_required": 8400,
                "small_k_execution_adaptation": False,
                "official_zero_step_due_to_drop_last_preserved": True,
            }
            if official_execution
            else None
        ),
        "artifacts": artifacts,
        "class_split_sha256": _sha256(split_path),
        "packages": packages,
        "cells": cells,
        "counts": {
            "packages": len(packages),
            "cells": len(cells),
            "scenario_rows": len(cells) * len(SCENARIOS),
        },
        "smoke_cell_ids": smoke_cell_ids,
        "smoke_receipt_sha256": None,
        "smoke_receipt_path": None,
        "launch_authority": False,
        "authority_state": "N607_PAPER_FULL_CI_SMOKE_REQUIRED",
    }
    expected_packages = len(receivers) * len(seeds) * len(new_counts)
    expected_cells = expected_packages * len(methods) * len(k_values)
    if plan["counts"] != {
        "packages": expected_packages,
        "cells": expected_cells,
        "scenario_rows": expected_cells * len(SCENARIOS),
    }:
        raise ValueError("paper-full matrix count drift")
    contract_payload = {
        key: value
        for key, value in plan.items()
        if key
        not in {
            "smoke_receipt_sha256",
            "smoke_receipt_path",
            "launch_authority",
            "authority_state",
            "plan_contract_sha256",
        }
    }
    plan["plan_contract_sha256"] = _canonical_sha256(contract_payload)
    if args.smoke_receipt:
        smoke_path = Path(args.smoke_receipt).resolve(strict=True)
        smoke = json.loads(smoke_path.read_text(encoding="utf-8-sig"))
        expected_artifact_hashes = {
            key: value["sha256"] for key, value in artifacts.items()
        }
        executed_plan_path = Path(
            str(smoke.get("executed_plan_path", ""))
        ).resolve(strict=True)
        predictor_sha256 = _sha256(
            Path(__file__).resolve().parents[2] / plan["predictor_script"]
        )
        if (
            smoke.get("schema")
            != "cvs.phase2.adv3b02_paper_full_ci_smoke_receipt.v1"
            or smoke.get("status") != "PASS"
            or smoke.get("completed_cell_ids") != smoke_cell_ids
            or smoke.get("executed_plan_sha256")
            != _sha256(executed_plan_path)
            or smoke.get("plan_contract_sha256")
            != plan["plan_contract_sha256"]
            or smoke.get("artifact_sha256") != expected_artifact_hashes
            or smoke.get("predictor_script_sha256") != predictor_sha256
        ):
            raise ValueError("smoke receipt does not authorize paper-full matrix")
        plan["smoke_receipt_sha256"] = _sha256(smoke_path)
        plan["smoke_receipt_path"] = str(smoke_path)
        plan["launch_authority"] = True
        plan["authority_state"] = "N607_PAPER_FULL_CI_SMOKE_PASS"
    _write_new(Path(args.output), plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="legacy pair or comma-separated nonempty official method subset",
    )
    parser.add_argument(
        "--new-counts",
        default=",".join(str(value) for value in NEW_COUNTS),
        help="ascending comma-separated nested new-class counts",
    )
    parser.add_argument("--required-total-capacity", type=int)
    parser.add_argument(
        "--expected-cache-scope",
        choices=("stage2_registered", "external_comparison_registered"),
        default="stage2_registered",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--target-cache-root", required=True)
    parser.add_argument("--cache-parity-root")
    parser.add_argument("--parity-reference-cache-root")
    parser.add_argument("--parity-preserved-class-labels", default="")
    parser.add_argument("--class-split", type=Path, required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-lock", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--head-artifact", required=True)
    parser.add_argument("--tta-policy", required=True)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, sort_keys=True))
