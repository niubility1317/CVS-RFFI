#!/usr/bin/env python3
"""Build the 100-package/800-cell paper-mechanism ADV3B02 CI plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


METHODS = ("csil_paper_full", "mopc_hr_paper_full")
NEW_COUNTS = (2, 5, 10, 20)
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    nested = {
        int(key): [str(item) for item in values]
        for key, values in split["nested_target_new_tx_labels"].items()
    }
    if tuple(sorted(nested)) != NEW_COUNTS:
        raise ValueError("new-class count matrix drift")
    if nested[20][:10] != nested[10] or nested[10][:5] != nested[5] or nested[5][:2] != nested[2]:
        raise ValueError("new-class sets are not nested")
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
            for new_count in NEW_COUNTS:
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
                        "predictor_package_root": str(package_parent / "predictor"),
                        "scorer_root": str(package_parent / "scorer"),
                        "detached_seal": str(package_parent / "predictor.seal.json"),
                        "build_receipt": str(package_parent / "package_build_receipt.json"),
                    }
                )
                for method in METHODS:
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
    smoke_cell_ids = [
        f"rx_20_1__seed_713101__new_2__{method}__k_1"
        for method in METHODS
    ] + [
        f"rx_20_1__seed_713101__new_20__{method}__k_20"
        for method in METHODS
    ]
    smoke_sha = None
    launch_authority = False
    authority_state = "N607_PAPER_FULL_CI_SMOKE_REQUIRED"
    if args.smoke_receipt:
        smoke_path = Path(args.smoke_receipt).resolve(strict=True)
        smoke = json.loads(smoke_path.read_text(encoding="utf-8-sig"))
        if (
            smoke.get("schema")
            != "cvs.phase2.adv3b02_paper_full_ci_smoke_receipt.v1"
            or smoke.get("status") != "PASS"
            or smoke.get("completed_cell_ids") != smoke_cell_ids
        ):
            raise ValueError("smoke receipt does not authorize paper-full matrix")
        smoke_sha = _sha256(smoke_path)
        launch_authority = True
        authority_state = "N607_PAPER_FULL_CI_SMOKE_PASS"
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
        "methods": list(METHODS),
        "receivers": receivers,
        "seeds": seeds,
        "k_values": k_values,
        "new_class_counts": list(NEW_COUNTS),
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
        "smoke_receipt_sha256": smoke_sha,
        "launch_authority": launch_authority,
        "authority_state": authority_state,
    }
    if plan["counts"] != {"packages": 100, "cells": 800, "scenario_rows": 2400}:
        raise ValueError("paper-full matrix count drift")
    _write_new(Path(args.output), plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--target-cache-root", required=True)
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
