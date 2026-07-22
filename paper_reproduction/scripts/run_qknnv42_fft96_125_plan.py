#!/usr/bin/env python3
"""Build, isolate, execute, and score the qKNNV42+FFT96 125-bundle plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
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
    result = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
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
    if plan.get("schema") != "cvs.phase2.qknnv42_fft96_125_plan.v1":
        raise ValueError("qKNNV42+FFT96 plan schema drift")
    if plan.get("counts") != {
        "packages": 100,
        "bundles": 125,
        "state_cells": 500,
        "scenario_rows": 1500,
    }:
        raise ValueError("qKNNV42+FFT96 plan count drift")
    return plan


def _selected(index: int, shard_index: int, shard_count: int) -> bool:
    return int(index) % int(shard_count) == int(shard_index)


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
            raise ValueError("existing qKNN package receipt is not PASS")
        return receipt
    parent = receipt_path.parent
    if parent.exists():
        raise RuntimeError("partial qKNN package parent exists; refusing destructive resume")
    parent.mkdir(parents=True, exist_ok=False)
    artifacts = plan["artifacts"]
    command = [
        sys.executable,
        str(project_root / "code/scripts/build_cvs_stage2_predictor_bundle.py"),
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
        package["stage"],
        "--receiver",
        package["receiver"],
        "--seed",
        str(package["seed"]),
        "--old-class-labels",
        ",".join(package["old_class_labels"]),
        "--new-class-count",
        str(package["new_class_count"]),
        "--support-pool-max-k",
        "20",
        "--query-per-tx",
        "20",
        "--candidate-lock",
        artifacts["candidate_lock"]["path"],
        "--checkpoint",
        artifacts["base_runtime"]["path"],
        "--adapter",
        artifacts["adapter"]["path"],
        "--head-artifact",
        artifacts["head_artifact"]["path"],
        "--tta-policy-json",
        artifacts["tta_policy"]["path"],
    ]
    if package["stage"] == "stage2c":
        command.extend(
            ["--new-class-labels", ",".join(package["new_class_labels"])]
        )
    else:
        command.extend(
            [
                "--stage2b-reference-new-class-labels",
                ",".join(package["reference_new_class_labels"]),
            ]
        )
    result = _run_json(command, cwd=project_root)
    receipt = {
        "schema": "cvs.phase2.qknnv42_fft96_package_build_receipt.v1",
        "status": "PASS",
        "package_id": package["package_id"],
        "stage": package["stage"],
        "registration_state": package["registration_state"],
        "source_free_repack": True,
        "source_adapter_or_head_copied": False,
        "reference_new_class_labels": list(
            package["reference_new_class_labels"]
        ),
        "reference_new_class_count": len(
            package["reference_new_class_labels"]
        ),
        "predictor_package_root_sha256": result["predictor_package_root_sha256"],
        "predictor_package_seal_sha256": result["predictor_package_seal_sha256"],
        "scoring_manifest_sha256": result["scoring_manifest_sha256"],
        "builder_result": result,
    }
    _write_new(receipt_path, receipt)
    return receipt


def _ensure_pre_run_evidence(
    package: dict[str, Any],
    receipt: dict[str, Any],
    *,
    project_root: Path,
    runtime_closure_root: Path,
    landlock_launcher: Path,
    landlock_policy_module: Path,
    strace: Path,
    python_executable: Path,
    system_read_roots: list[Path],
) -> Path:
    evidence_root = Path(package["pre_run_evidence_root"])
    evidence_json = evidence_root / "runtime_isolation_evidence.json"
    if evidence_json.is_file():
        return evidence_json
    if evidence_root.exists():
        raise RuntimeError("partial pre-run evidence root exists")
    command = [
        sys.executable,
        str(
            project_root
            / "code/scripts/build_cvs_stage2_landlock_pre_run_evidence.py"
        ),
        "--runtime-closure-root",
        str(runtime_closure_root),
        "--predictor-package-root",
        package["predictor_package_root"],
        "--detached-seal-path",
        package["detached_seal"],
        "--expected-package-seal-sha256",
        receipt["predictor_package_seal_sha256"],
        "--output-root",
        str(evidence_root),
        "--landlock-launcher",
        str(landlock_launcher),
        "--landlock-policy-module",
        str(landlock_policy_module),
        "--strace-executable",
        str(strace),
        "--python-executable",
        str(python_executable),
        "--forbidden-scorer-truth-root",
        package["scorer_root"],
    ]
    for root in system_read_roots:
        command.extend(["--system-read-root", str(root)])
    _run_json(command, cwd=project_root)
    return evidence_json


def _load_formal_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != "cvs.phase2.formal_metric_rows.v1":
        raise ValueError("formal metric row schema drift")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("one state cell must produce three scenario rows")
    return rows


def _run_state_cell(
    plan: dict[str, Any],
    cell: dict[str, Any],
    package: dict[str, Any],
    *,
    project_root: Path,
    runtime_closure_root: Path,
    landlock_launcher: Path,
    landlock_policy_module: Path,
    strace: Path,
    python_executable: Path,
    system_read_roots: list[Path],
    gpu_devices: list[Path],
    device: str,
    forbidden_project_root: str,
) -> dict[str, Any]:
    output_root = Path(cell["output_root"])
    cell_receipt_path = output_root / "cell_receipt.json"
    if cell_receipt_path.is_file():
        receipt = json.loads(cell_receipt_path.read_text(encoding="utf-8-sig"))
        if receipt.get("status") != "LOCAL_DIAGNOSTIC_PASS":
            raise ValueError("existing state-cell receipt is not diagnostic PASS")
        return receipt
    if output_root.exists():
        raise RuntimeError("partial state-cell output exists; refusing destructive resume")
    package_receipt = _build_package(plan, package, project_root=project_root)
    evidence_json = _ensure_pre_run_evidence(
        package,
        package_receipt,
        project_root=project_root,
        runtime_closure_root=runtime_closure_root,
        landlock_launcher=landlock_launcher,
        landlock_policy_module=landlock_policy_module,
        strace=strace,
        python_executable=python_executable,
        system_read_roots=system_read_roots,
    )
    request_root = output_root / "request"
    predictor_root = output_root / "predictor"
    scoring_root = output_root / "scoring"
    request_root.mkdir(parents=True, exist_ok=False)
    predictor_root.mkdir(parents=True, exist_ok=False)
    request_json = request_root / "request.json"
    request_result = _run_json(
        [
            sys.executable,
            str(project_root / "code/scripts/build_cvs_stage2_predictor_request.py"),
            "--predictor-package-root",
            package["predictor_package_root"],
            "--detached-seal-path",
            package["detached_seal"],
            "--expected-seal-sha256",
            package_receipt["predictor_package_seal_sha256"],
            "--runtime-evidence-json",
            str(evidence_json),
            "--k-shot",
            str(cell["k_shot"]),
            "--request-id",
            f"req_{hashlib.sha256(cell['cell_id'].encode('utf-8')).hexdigest()}",
            "--row-id",
            f"row_{hashlib.sha256(cell['cell_id'].encode('utf-8')).hexdigest()}",
            "--output-json",
            str(request_json),
        ],
        cwd=project_root,
    )
    isolated_command = [
        sys.executable,
        str(project_root / "code/scripts/run_cvs_stage2_landlock_isolated.py"),
        "--landlock-launcher",
        str(landlock_launcher),
        "--landlock-policy-module",
        str(landlock_policy_module),
        "--strace",
        str(strace),
        "--runtime-closure-root",
        str(runtime_closure_root),
        "--pre-run-evidence-root",
        package["pre_run_evidence_root"],
        "--predictor-package-root",
        package["predictor_package_root"],
        "--detached-seal-path",
        package["detached_seal"],
        "--expected-package-seal-sha256",
        package_receipt["predictor_package_seal_sha256"],
        "--request-json",
        str(request_json),
        "--output-root",
        str(predictor_root),
        "--python-executable",
        str(python_executable),
        "--forbidden-root",
        package["scorer_root"],
        "--forbidden-project-root",
        forbidden_project_root,
        "--device",
        device,
        "--batch-size",
        "256",
    ]
    for root in system_read_roots:
        isolated_command.extend(["--system-read-root", str(root)])
    for gpu in gpu_devices:
        isolated_command.extend(["--gpu-device", str(gpu)])
    isolated = _run_json(isolated_command, cwd=project_root)
    if isolated.get("status") != "LOCAL_DIAGNOSTIC_PASS":
        raise ValueError("isolated qKNN predictor did not pass its real access ledger")
    stdout_receipt = json.loads(
        Path(isolated["predictor_stdout_receipt"]).read_text(encoding="utf-8-sig")
    )
    predictor_result = stdout_receipt.get("predictor_result")
    if not isinstance(predictor_result, dict):
        raise ValueError("isolated predictor stdout lacks result")
    scoring_root.mkdir(parents=True, exist_ok=False)
    scoring_manifest = Path(package["scorer_root"]) / "scoring_manifest.json"
    scoring = _run_json(
        [
            sys.executable,
            str(project_root / "code/scripts/score_cvs_stage2_sealed_prediction.py"),
            "--prediction-artifact",
            str(predictor_root / "prediction_artifact.cvspred"),
            "--expected-prediction-artifact-sha256",
            predictor_result["artifact_sha256"],
            "--expected-prediction-seal-sha256",
            predictor_result["seal_sha256"],
            "--scoring-manifest",
            str(scoring_manifest),
            "--expected-scoring-manifest-sha256",
            package_receipt["scoring_manifest_sha256"],
            "--formal-rows",
            str(scoring_root / "formal_rows.json"),
            "--formal-predictions",
            str(scoring_root / "formal_predictions.json"),
            "--scoring-receipt",
            str(scoring_root / "scoring_receipt.json"),
        ],
        cwd=project_root,
    )
    rows = _load_formal_rows(scoring_root / "formal_rows.json")
    receipt = {
        "schema": "cvs.phase2.qknnv42_fft96_state_cell_receipt.v1",
        "status": "LOCAL_DIAGNOSTIC_PASS",
        **{
            key: cell[key]
            for key in (
                "cell_id",
                "bundle_id",
                "package_id",
                "registration_state",
                "receiver",
                "seed",
                "seed_role",
                "k_shot",
                "new_class_count",
            )
        },
        "output_root": str(output_root),
        "request_sha256": request_result["request_sha256"],
        "predictor_package_root_sha256": package_receipt[
            "predictor_package_root_sha256"
        ],
        "predictor_package_seal_sha256": package_receipt[
            "predictor_package_seal_sha256"
        ],
        "prediction_artifact_sha256": predictor_result["artifact_sha256"],
        "prediction_seal_sha256": predictor_result["seal_sha256"],
        "filesystem_access_audit_sha256": isolated[
            "filesystem_access_audit_sha256"
        ],
        "scoring_receipt_sha256": _sha256(scoring_root / "scoring_receipt.json"),
        "formal_rows_sha256": _sha256(scoring_root / "formal_rows.json"),
        "scenario_row_count": len(rows),
        "scoring_status": scoring.get("status"),
        "formal_launch_authority": False,
        "protocol_valid_claim_allowed": False,
    }
    _write_new(cell_receipt_path, receipt)
    return receipt


def _write_smoke_receipt(
    plan: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        "before_registration": {0},
        "after_registration": {5, 10, 20},
    }
    actual: dict[str, set[int]] = {}
    for receipt in receipts:
        actual.setdefault(receipt["registration_state"], set()).add(
            int(receipt["new_class_count"])
        )
    if actual != expected:
        raise ValueError("smoke bundle does not contain all four physical states")
    payload = {
        "schema": "cvs.phase2.qknnv42_fft96_real_smoke_receipt.v1",
        "status": "PASS",
        "bundle_id": plan["smoke_bundle_id"],
        "real_landlock_seccomp_strace_state_cell_count": len(receipts),
        "all_filesystem_access_ledgers_pass": True,
        "state_cell_receipt_sha256": {
            receipt["cell_id"]: _sha256(
                Path(receipt["output_root"]) / "cell_receipt.json"
            )
            for receipt in receipts
        },
        "matrix_launch_allowed_as_diagnostic_only": True,
        "formal_launch_authority": False,
        "protocol_valid_claim_allowed": False,
    }
    target = Path(plan["run_root"]) / "smoke_receipt.json"
    _write_new(target, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).resolve(strict=True)
    plan = _load_plan(plan_path)
    project_root = Path(args.project_root).resolve(strict=True)
    runtime_closure_root = Path(args.runtime_closure_root).resolve(strict=True)
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index/count")
    packages = {item["package_id"]: item for item in plan["packages"]}
    system_read_roots = [Path(value).resolve(strict=True) for value in args.system_read_root]
    gpu_devices = [Path(value).resolve(strict=True) for value in args.gpu_device if Path(value).exists()]
    common = {
        "project_root": project_root,
        "runtime_closure_root": runtime_closure_root,
        "landlock_launcher": Path(args.landlock_launcher).resolve(strict=True),
        "landlock_policy_module": Path(
            args.landlock_policy_module
        ).resolve(strict=True),
        "strace": Path(args.strace).resolve(strict=True),
        "python_executable": Path(args.python_executable).resolve(strict=True),
        "system_read_roots": system_read_roots,
        "gpu_devices": gpu_devices,
        "device": args.device,
        "forbidden_project_root": str(project_root),
    }
    completed: list[str] = []
    if args.stage == "build_shard":
        for index, package in enumerate(plan["packages"]):
            if _selected(index, args.shard_index, args.shard_count):
                receipt = _build_package(plan, package, project_root=project_root)
                _ensure_pre_run_evidence(
                    package,
                    receipt,
                    project_root=project_root,
                    runtime_closure_root=runtime_closure_root,
                    landlock_launcher=common["landlock_launcher"],
                    landlock_policy_module=common[
                        "landlock_policy_module"
                    ],
                    strace=common["strace"],
                    python_executable=common["python_executable"],
                    system_read_roots=system_read_roots,
                )
                completed.append(package["package_id"])
    elif args.stage == "smoke":
        if int(args.shard_index) != 0:
            raise ValueError("smoke runs only on shard 0")
        cells = [
            item
            for item in plan["state_cells"]
            if item["bundle_id"] == plan["smoke_bundle_id"]
        ]
        receipts = [
            _run_state_cell(
                plan,
                cell,
                packages[cell["package_id"]],
                **common,
            )
            for cell in cells
        ]
        _write_smoke_receipt(plan, receipts)
        completed.extend(item["cell_id"] for item in receipts)
    else:
        smoke = Path(args.smoke_receipt or Path(plan["run_root"]) / "smoke_receipt.json")
        if not smoke.is_file():
            raise ValueError("matrix launch requires the real N607 smoke receipt")
        smoke_payload = json.loads(smoke.read_text(encoding="utf-8-sig"))
        if (
            smoke_payload.get("status") != "PASS"
            or smoke_payload.get("matrix_launch_allowed_as_diagnostic_only") is not True
            or smoke_payload.get("formal_launch_authority") is not False
        ):
            raise ValueError("smoke receipt does not authorize diagnostic matrix launch")
        for index, cell in enumerate(plan["state_cells"]):
            if _selected(index, args.shard_index, args.shard_count):
                completed.append(
                    _run_state_cell(
                        plan,
                        cell,
                        packages[cell["package_id"]],
                        **common,
                    )["cell_id"]
                )
    return {
        "status": "PASS",
        "stage": args.stage,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "completed": completed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("build_shard", "smoke", "matrix_shard"),
        required=True,
    )
    parser.add_argument("--runtime-closure-root", type=Path, required=True)
    parser.add_argument("--landlock-launcher", type=Path, required=True)
    parser.add_argument("--landlock-policy-module", type=Path, required=True)
    parser.add_argument("--strace", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--system-read-root", action="append", required=True)
    parser.add_argument("--gpu-device", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--smoke-receipt", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
