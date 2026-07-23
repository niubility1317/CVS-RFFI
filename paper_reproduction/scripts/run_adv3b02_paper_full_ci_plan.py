#!/usr/bin/env python3
"""Build packages and run the sharded paper-mechanism ADV3B02 CI plan."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("csil_paper_full", "mopc_hr_paper_full")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_json(command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("command returned no JSON")
    return json.loads(lines[-1])


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "cvs.phase2.adv3b02_paper_full_ci_plan.v1":
        raise ValueError("paper-full plan schema drift")
    if tuple(plan.get("methods", [])) != METHODS:
        raise ValueError("paper-full methods drift")
    if plan.get("counts") != {"packages": 100, "cells": 800, "scenario_rows": 2400}:
        raise ValueError("paper-full matrix counts drift")
    return plan


def _selected(index: int, shard_index: int, shard_count: int) -> bool:
    return int(index) % int(shard_count) == int(shard_index)


def _load_formal_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != "cvs.phase2.formal_metric_rows.v1":
        raise ValueError("formal metric row schema drift")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("formal metric rows payload drift")
    return rows


def _build_package(
    plan: dict[str, Any],
    package: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    receipt_path = Path(package["build_receipt"])
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        if receipt.get("status") != "PASS":
            raise ValueError("existing package receipt is not PASS")
        return receipt
    if receipt_path.parent.exists():
        raise RuntimeError("partial package parent exists; refusing destructive resume")
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    artifacts = plan["artifacts"]
    result = _run_json(
        [
            sys.executable,
            str(
                project_root
                / "paper_reproduction/scripts/build_adv3b02_paper_full_ci_bundle.py"
            ),
            "--target-cache-set",
            package["target_cache_set"],
            "--expected-cache-scope",
            "stage2_registered",
            "--predictor-out-root",
            package["predictor_package_root"],
            "--scorer-out-root",
            package["scorer_root"],
            "--detached-seal-path",
            package["detached_seal"],
            "--stage",
            "stage2c",
            "--receiver",
            package["receiver"],
            "--seed",
            str(package["seed"]),
            "--old-class-labels",
            ",".join(package["old_class_labels"]),
            "--new-class-labels",
            ",".join(package["new_class_labels"]),
            "--new-class-count",
            str(package["new_class_count"]),
            "--support-pool-max-k",
            "20",
            "--query-per-tx",
            "20",
            "--candidate-lock",
            artifacts["candidate_lock"]["path"],
            "--checkpoint",
            artifacts["base_checkpoint"]["path"],
            "--adapter",
            artifacts["adapter"]["path"],
            "--head-artifact",
            artifacts["head_artifact"]["path"],
            "--tta-policy-json",
            artifacts["tta_policy"]["path"],
        ],
        cwd=project_root,
    )
    receipt = {
        "schema": "cvs.phase2.adv3b02_paper_full_ci_package_build_receipt.v1",
        "status": "PASS",
        "package_id": package["package_id"],
        "predictor_package_root_sha256": result["predictor_package_root_sha256"],
        "predictor_package_seal_sha256": result["predictor_package_seal_sha256"],
        "scoring_manifest_sha256": result["scoring_manifest_sha256"],
        "builder_result": result,
    }
    _write_new(receipt_path, receipt)
    return receipt


def _run_cell(
    plan: dict[str, Any],
    cell: dict[str, Any],
    package: dict[str, Any],
    *,
    project_root: Path,
    device: str,
) -> dict[str, Any]:
    output_root = Path(cell["output_root"])
    cell_receipt = output_root / "cell_receipt.json"
    if cell_receipt.is_file():
        value = json.loads(cell_receipt.read_text(encoding="utf-8-sig"))
        if value.get("status") != "FORMAL_COMPARISON_BASELINE":
            raise ValueError("existing paper-full cell receipt status drift")
        return value
    if output_root.exists():
        raise RuntimeError("partial cell output exists; refusing destructive resume")
    build_receipt = _build_package(plan, package, project_root=project_root)
    predictor_root = output_root / "predictor"
    predictor = _run_json(
        [
            sys.executable,
            str(project_root / plan["predictor_script"]),
            "--package-root",
            package["predictor_package_root"],
            "--detached-seal",
            package["detached_seal"],
            "--expected-seal-sha256",
            build_receipt["predictor_package_seal_sha256"],
            "--method",
            cell["method"],
            "--old-class-count",
            "6",
            "--k-shot",
            str(cell["k_shot"]),
            "--seed",
            str(cell["seed"]),
            "--row-id",
            cell["cell_id"],
            "--output-dir",
            str(predictor_root),
            "--device",
            device,
        ],
        cwd=project_root,
    )
    if predictor.get("status") != "FORMAL_COMPARISON_BASELINE":
        raise ValueError("paper-full predictor status drift")
    scoring_root = output_root / "scoring"
    scoring_root.mkdir(parents=True, exist_ok=False)
    scoring_manifest = Path(package["scorer_root"]) / "scoring_manifest.json"
    scoring = _run_json(
        [
            sys.executable,
            str(project_root / "code/scripts/score_cvs_stage2_sealed_prediction.py"),
            "--prediction-artifact",
            predictor["prediction_artifact"],
            "--expected-prediction-artifact-sha256",
            predictor["prediction_artifact_sha256"],
            "--expected-prediction-seal-sha256",
            predictor["prediction_seal_sha256"],
            "--scoring-manifest",
            str(scoring_manifest),
            "--expected-scoring-manifest-sha256",
            build_receipt["scoring_manifest_sha256"],
            "--formal-rows",
            str(scoring_root / "formal_rows.json"),
            "--formal-predictions",
            str(scoring_root / "formal_predictions.json"),
            "--scoring-receipt",
            str(scoring_root / "scoring_receipt.json"),
        ],
        cwd=project_root,
    )
    if len(_load_formal_rows(scoring_root / "formal_rows.json")) != 3:
        raise ValueError("cell scorer did not produce three scenario rows")
    receipt = {
        "schema": "cvs.phase2.adv3b02_paper_full_ci_cell_receipt.v1",
        "status": "FORMAL_COMPARISON_BASELINE",
        **{
            key: cell[key]
            for key in (
                "cell_id",
                "package_id",
                "receiver",
                "seed",
                "new_class_count",
                "method",
                "k_shot",
            )
        },
        "predictor_receipt_sha256": predictor["predictor_receipt_sha256"],
        "prediction_artifact_sha256": predictor["prediction_artifact_sha256"],
        "prediction_seal_sha256": predictor["prediction_seal_sha256"],
        "scoring_receipt_sha256": _sha256(scoring_root / "scoring_receipt.json"),
        "formal_rows_sha256": _sha256(scoring_root / "formal_rows.json"),
        "scoring_status": scoring.get("status"),
    }
    _write_new(cell_receipt, receipt)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(Path(args.plan).resolve(strict=True))
    project_root = Path(args.project_root).resolve(strict=True)
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index/count")
    packages = {item["package_id"]: item for item in plan["packages"]}
    cells = {item["cell_id"]: item for item in plan["cells"]}
    completed = []
    if args.stage == "build_shard":
        for index, package in enumerate(plan["packages"]):
            if _selected(index, args.shard_index, args.shard_count):
                completed.append(
                    _build_package(plan, package, project_root=project_root)["package_id"]
                )
    elif args.stage == "smoke":
        if int(args.shard_index) != 0:
            raise ValueError("smoke runs only on shard 0")
        for cell_id in plan["smoke_cell_ids"]:
            cell = cells[cell_id]
            _run_cell(
                plan,
                cell,
                packages[cell["package_id"]],
                project_root=project_root,
                device=args.device,
            )
            completed.append(cell_id)
        receipt = {
            "schema": "cvs.phase2.adv3b02_paper_full_ci_smoke_receipt.v1",
            "status": "PASS",
            "completed_cell_ids": completed,
            "cell_receipt_sha256": {
                cell_id: _sha256(Path(cells[cell_id]["output_root"]) / "cell_receipt.json")
                for cell_id in completed
            },
        }
        _write_new(Path(plan["run_root"]) / "smoke_receipt.json", receipt)
    else:
        if (
            plan.get("launch_authority") is not True
            or plan.get("authority_state") != "N607_PAPER_FULL_CI_SMOKE_PASS"
        ):
            raise ValueError("formal paper-full matrix lacks smoke authority")
        cells_by_package = {}
        for cell in plan["cells"]:
            cells_by_package.setdefault(cell["package_id"], []).append(cell)
        for index, package in enumerate(plan["packages"]):
            if not _selected(index, args.shard_index, args.shard_count):
                continue
            for cell in cells_by_package[package["package_id"]]:
                completed.append(
                    _run_cell(
                        plan,
                        cell,
                        package,
                        project_root=project_root,
                        device=args.device,
                    )["cell_id"]
                )
    return {
        "status": "PASS",
        "stage": args.stage,
        "shard_index": args.shard_index,
        "completed": completed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("build_shard", "smoke", "matrix_shard"), required=True
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
