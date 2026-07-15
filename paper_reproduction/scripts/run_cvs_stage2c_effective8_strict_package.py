#!/usr/bin/env python3
"""Build and execute one strict effective8 package for its locked K-shot cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    preflight_stage2_predictor_package,
    sha256_file,
)
from cvsrffi.stage2_scoring_sidecar import load_verified_scoring_sidecar  # noqa: E402
from paper_reproduction.scripts.build_cvs_stage2c_effective8_strict_plan import (  # noqa: E402
    validate_strict_plan,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    if os.name == "posix":
        path.chmod(0o444)


def _run_json(command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"strict command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"stdout={completed.stdout[-4000:]}\nstderr={completed.stderr[-4000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"strict command returned no JSON: {' '.join(command)}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise RuntimeError("strict command JSON result must be an object")
    return value


def _package_by_id(plan: Mapping[str, Any], package_id: str) -> dict[str, Any]:
    matched = [item for item in plan["package_steps"] if item["package_id"] == package_id]
    if len(matched) != 1:
        raise ValueError(f"strict package id is absent or duplicated: {package_id}")
    return dict(matched[0])


def package_build_command(
    plan: Mapping[str, Any], package: Mapping[str, Any]
) -> list[str]:
    artifacts = dict(plan["runtime_artifacts"])
    return [
        sys.executable,
        str(REPO_ROOT / "code/scripts/build_cvs_stage2_predictor_bundle.py"),
        "--target-cache-set", str(package["target_cache_set"]),
        "--predictor-out-root", str(package["predictor_package_root"]),
        "--scorer-out-root", str(package["scorer_root"]),
        "--detached-seal-path", str(package["detached_seal"]),
        "--stage", "stage2c",
        "--receiver", str(package["receiver"]),
        "--seed", str(package["seed"]),
        "--old-class-labels", ",".join(package["old_class_labels"]),
        "--new-class-labels", ",".join(package["new_class_labels"]),
        "--new-class-count", str(package["new_class_count"]),
        "--support-pool-max-k", str(plan["support_pool_max_k"]),
        "--query-per-tx", str(plan["query_per_tx"]),
        "--candidate-lock", artifacts["candidate_lock"],
        "--checkpoint", artifacts["candidate_runtime"],
        "--base-checkpoint", artifacts["base_runtime"],
        "--candidate-capsule", artifacts["candidate_capsule"],
        "--expected-candidate-capsule-sha256", plan["candidate_capsule_sha256"],
        "--runtime-config-receipt", artifacts["runtime_config_receipt"],
        "--adapter", artifacts["runtime_adapter"],
        "--head-artifact", artifacts["runtime_head"],
        "--tta-policy-json", artifacts["runtime_tta"],
    ]


def _ensure_package(
    plan: Mapping[str, Any], package: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    package_root = Path(package["predictor_package_root"])
    scorer_root = Path(package["scorer_root"])
    seal = Path(package["detached_seal"])
    present = (package_root.exists(), scorer_root.exists(), seal.exists())
    if any(present) and not all(present):
        raise RuntimeError("partial strict package exists; refusing destructive resume")
    if not all(present):
        result = _run_json(package_build_command(plan, package), cwd=project_root)
        expected_seal = str(result["predictor_package_seal_sha256"])
    else:
        expected_seal = sha256_file(seal)
    manifest, _seal, audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=seal,
        expected_seal_sha256=expected_seal,
    )
    scoring_path = scorer_root / "scoring_manifest.json"
    _truth, scoring, _scoring_audit = load_verified_scoring_sidecar(scoring_path)
    if (
        audit.get("status") != "PASS"
        or scoring["predictor_package_root_sha256"] != manifest["package_root_sha256"]
        or scoring["predictor_package_seal_sha256"] != expected_seal
    ):
        raise RuntimeError("strict predictor/scorer package binding failed")
    return {
        "package_manifest": manifest,
        "package_seal_sha256": expected_seal,
        "scoring_manifest": scoring_path,
        "scoring_manifest_sha256": sha256_file(scoring_path),
    }


def _cell_by_k(package: Mapping[str, Any], k_shot: int) -> dict[str, Any]:
    matched = [cell for cell in package["cells"] if int(cell["k_shot"]) == int(k_shot)]
    if len(matched) != 1:
        raise ValueError(f"strict package K-shot cell is absent or duplicated: K={k_shot}")
    return dict(matched[0])


def _execute_cell(
    plan: Mapping[str, Any],
    package: Mapping[str, Any],
    cell: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    project_root: Path,
    device: str,
) -> dict[str, Any]:
    receipt_path = Path(cell["cell_receipt"])
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if receipt.get("status") != "PROTOCOL_VALID" or receipt.get("cell_id") != cell["cell_id"]:
            raise RuntimeError("existing strict cell receipt is not reusable")
        return receipt
    evidence_root = Path(cell["pre_run_evidence_root"])
    request_path = Path(cell["request_json"])
    output_root = Path(cell["predictor_output_root"])
    scoring_output = Path(cell["scoring_output_root"])
    if any(path.exists() for path in (evidence_root, request_path, output_root, scoring_output)):
        raise RuntimeError("partial strict cell exists; refusing destructive resume")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = dict(plan["runtime_artifacts"])
    seal_sha = str(binding["package_seal_sha256"])
    _run_json(
        [
            sys.executable,
            str(REPO_ROOT / "code/scripts/build_cvs_phase2_landlock_pre_run_evidence.py"),
            "--runtime-closure-root", artifacts["runtime_closure_root"],
            "--package-root", str(package["predictor_package_root"]),
            "--detached-seal", str(package["detached_seal"]),
            "--expected-package-seal-sha256", seal_sha,
            "--scorer-root", str(package["scorer_root"]),
            "--candidate-capsule", artifacts["candidate_capsule"],
            "--expected-candidate-capsule-sha256", str(plan["candidate_capsule_sha256"]),
            "--runtime-config-receipt", artifacts["runtime_config_receipt"],
            "--python-executable", sys.executable,
            "--strace-executable", str(plan["strace_executable"]),
            "--landlock-launcher", str(plan["landlock_launcher"]),
            "--output-root", str(evidence_root),
        ],
        cwd=project_root,
    )
    _run_json(
        [
            sys.executable,
            str(REPO_ROOT / "code/scripts/build_cvs_stage2_predictor_request.py"),
            "--predictor-package-root", str(package["predictor_package_root"]),
            "--detached-seal-path", str(package["detached_seal"]),
            "--expected-seal-sha256", seal_sha,
            "--runtime-evidence-json", str(evidence_root / "runtime_isolation_evidence.json"),
            "--k-shot", str(cell["k_shot"]),
            "--request-id", str(cell["cell_id"]),
            "--row-id", str(cell["cell_id"]),
            "--output-json", str(request_path),
        ],
        cwd=project_root,
    )
    output_root.mkdir(parents=False, exist_ok=False)
    runner = _run_json(
        [
            sys.executable,
            str(REPO_ROOT / "code/scripts/run_cvs_stage2_landlock_pinned.py"),
            "--runtime-closure-root", artifacts["runtime_closure_root"],
            "--pre-run-evidence-root", str(evidence_root),
            "--package-root", str(package["predictor_package_root"]),
            "--detached-seal", str(package["detached_seal"]),
            "--expected-package-seal-sha256", seal_sha,
            "--request-json", str(request_path),
            "--output-root", str(output_root),
            "--python-executable", sys.executable,
            "--strace-executable", str(plan["strace_executable"]),
            "--landlock-launcher", str(plan["landlock_launcher"]),
            "--forbidden-root", str(plan["target_dataset_forbidden_root"]),
            "--forbidden-root", str(Path(package["target_cache_set"]).parent),
            "--forbidden-root", str(package["scorer_root"]),
            "--device", device,
        ],
        cwd=project_root,
    )
    if runner.get("status") != "PROTOCOL_VALID" or runner.get("formal_launch_authority") is not True:
        raise RuntimeError("strict Landlock predictor did not return PROTOCOL_VALID")
    scoring_output.mkdir(parents=False, exist_ok=False)
    prediction = output_root / "prediction_artifact.cvspred"
    scoring = _run_json(
        [
            sys.executable,
            str(REPO_ROOT / "code/scripts/score_cvs_stage2_sealed_prediction.py"),
            "--prediction-artifact", str(prediction),
            "--expected-prediction-artifact-sha256", str(runner["prediction_artifact_sha256"]),
            "--expected-prediction-seal-sha256", str(runner["prediction_seal_sha256"]),
            "--scoring-manifest", str(binding["scoring_manifest"]),
            "--expected-scoring-manifest-sha256", str(binding["scoring_manifest_sha256"]),
            "--formal-rows", str(scoring_output / "formal_rows.json"),
            "--formal-predictions", str(scoring_output / "formal_predictions.json"),
            "--scoring-receipt", str(scoring_output / "scoring_receipt.json"),
        ],
        cwd=project_root,
    )
    if int(scoring.get("scenario_count", -1)) != 3 or int(scoring.get("formal_row_count", -1)) != 3:
        raise RuntimeError("strict scorer did not produce exactly three formal scenario rows")
    receipt = {
        "schema": "cvs.stage2c.effective8.strict_cell_receipt.v1",
        "status": "PROTOCOL_VALID",
        "cell_id": cell["cell_id"],
        "package_id": package["package_id"],
        "receiver": package["receiver"],
        "seed": package["seed"],
        "new_class_count": package["new_class_count"],
        "k_shot": cell["k_shot"],
        "candidate_capsule_sha256": plan["candidate_capsule_sha256"],
        "package_seal_sha256": seal_sha,
        "prediction_artifact_sha256": runner["prediction_artifact_sha256"],
        "prediction_seal_sha256": runner["prediction_seal_sha256"],
        "formal_post_run_runtime_evidence_sha256": runner["formal_post_run_runtime_evidence_sha256"],
        "scoring_receipt_sha256": sha256_file(scoring_output / "scoring_receipt.json"),
        "formal_scenario_row_count": 3,
    }
    _write_new(receipt_path, receipt)
    return receipt


def run_package(
    plan: Mapping[str, Any],
    *,
    package_id: str,
    project_root: Path,
    device: str,
    k_values: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    plan = validate_strict_plan(plan)
    package = _package_by_id(plan, package_id)
    binding = _ensure_package(plan, package, project_root=project_root)
    selected = tuple(int(value) for value in (k_values or [cell["k_shot"] for cell in package["cells"]]))
    return [
        _execute_cell(
            plan,
            package,
            _cell_by_k(package, k_shot),
            binding,
            project_root=project_root,
            device=device,
        )
        for k_shot in selected
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k-shot", type=int, action="append")
    args = parser.parse_args(argv)
    receipts = run_package(
        _read_json(args.plan_manifest),
        package_id=args.package_id,
        project_root=args.project_root.resolve(strict=True),
        device=args.device,
        k_values=args.k_shot,
    )
    print(json.dumps({"status": "PROTOCOL_VALID", "receipts": receipts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
