#!/usr/bin/env python
"""Execute the locked JG_R8_LR020 K10 Stage2-C development sequence.

Run this only after the N607 preflight and GPU inventory have been recorded.
The launcher is sequential and creates every output exclusively.  Its three
rows repeat the identical old-only adapter fit for row-level isolation; this
is experimental repeated compute, not required deployment compute.  A real
5->10->20 deployment trains JG once, freezes it, then appends prototypes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "code"
for value in (str(REPO), str(CODE)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO), str(CODE)):
    sys.path.insert(0, value)

from paper_reproduction.cvs_aligned.jg020_stage2c import (  # noqa: E402
    sha256_file,
    validate_locked_candidate,
)


EXPERIMENT_ID = "qknnv42_jg_r8_lr020_newclass_dev_20260716"
REMOTE_REPO = Path("/home/szu2070436088/2510044040/CV-SincNet")
DEFAULT_RUN_ROOT = REMOTE_REPO / "runs" / EXPERIMENT_ID
RETRY1_RUN_ROOT = REMOTE_REPO / "runs" / f"{EXPERIMENT_ID}_retry1"
RETRY2_RUN_ROOT = REMOTE_REPO / "runs" / f"{EXPERIMENT_ID}_retry2"
ORIGINAL_CACHE_SET = DEFAULT_RUN_ROOT / "phase1_cache/cache_set.json"
CHECKPOINT = (
    REMOTE_REPO
    / "runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200"
    / "best_joint_safe_ssdg.pth"
)
GROUND_P4 = (
    REMOTE_REPO
    / "runs/qknn_ground_adapt_layer_loss_ablation_20260715_v16/p4_r16_e8_k1"
    / "adapter_fp16.pt"
)
MAPPING = REPO / "paper_reproduction/configs/adv3b02_target_old_class_mapping_20260715.json"
CACHE_SPEC = REPO / "paper_reproduction/configs/jg020_stage2c_registered_cache_rx20_1_seed713101_20260716.json"
SOURCE_ADAPTER = REPO / "paper_reproduction/configs/jg020_offline_source_adapter_descriptor_20260716.json"
SOURCE_HEAD = REPO / "paper_reproduction/configs/jg020_offline_source_head_descriptor_20260716.json"
VIEW_POLICY = REPO / "paper_reproduction/configs/jg020_single_view_policy_20260716.json"
OLD_LABELS = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
NEW_LABELS = [
    "1-16", "1-18", "18-10", "14-11", "8-3", "18-8", "10-10",
    "16-19", "20-12", "4-10", "13-14", "2-5", "1-8", "19-13",
    "19-9", "3-8", "19-8", "11-19", "2-16", "19-6",
]


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _command_plan(run_root: Path, python: str, *, cache_set: Path | None = None) -> list[dict[str, Any]]:
    cache_set = cache_set or run_root / "phase1_cache/cache_set.json"
    if cache_set == run_root / "phase1_cache/cache_set.json":
        plan: list[dict[str, Any]] = [
            {
                "phase": "phase1_offline_cache",
                "command": [python, "code/scripts/build_cvs_leo_weak_iq_cache.py", "--spec", str(CACHE_SPEC), "--device", "cuda:0"],
            }
        ]
    else:
        plan = [{"phase": "reuse_sealed_phase1_cache", "cache_set": str(cache_set)}]
    for count in (5, 10, 20):
        row = run_root / f"new_{count}"
        lock = REPO / f"paper_reproduction/configs/jg020_stage2c_candidate_lock_new{count}_20260716.json"
        source = row / "source_package"
        source_scorer = row / "source_scorer"
        source_seal = row / "source_package.seal.json"
        enrollment = row / "enrollment_package"
        enrollment_seal = row / "enrollment_package.seal.json"
        state = row / "enrollment_state"
        apply_root = row / "apply_package"
        apply_seal = row / "apply_package.seal.json"
        scorer = row / "scorer"
        predictions = row / "predictor_output"
        scores = row / "scores"
        plan.extend(
            [
                {
                    "phase": f"new_{count}:offline_source_bundle",
                    "command": [
                        python, "code/scripts/build_cvs_stage2_predictor_bundle.py",
                        "--target-cache-set", str(cache_set), "--expected-cache-scope", "stage2_registered",
                        "--predictor-out-root", str(source), "--scorer-out-root", str(source_scorer),
                        "--detached-seal-path", str(source_seal), "--stage", "stage2c",
                        "--receiver", "20-1", "--seed", "713101",
                        "--old-class-labels", ",".join(OLD_LABELS),
                        "--new-class-labels", ",".join(NEW_LABELS[:count]),
                        "--new-class-count", str(count), "--support-pool-max-k", "10",
                        "--query-per-tx", "20", "--candidate-lock", str(lock),
                        "--checkpoint", str(CHECKPOINT), "--adapter", str(SOURCE_ADAPTER),
                        "--head-artifact", str(SOURCE_HEAD), "--tta-policy-json", str(VIEW_POLICY),
                    ],
                },
                {
                    "phase": f"new_{count}:split_enrollment",
                    "dynamic_hash_inputs": [str(source_seal)],
                    "command_prefix": [
                        python, "paper_reproduction/scripts/build_cvs_jg020_split_packages.py", "enrollment",
                        "--source-package-root", str(source), "--source-detached-seal", str(source_seal),
                    ],
                },
                {"phase": f"new_{count}:support_only_enroll", "dynamic_hash_inputs": [str(enrollment_seal)]},
                {"phase": f"new_{count}:split_apply", "dynamic_hash_inputs": [str(source_seal), str(enrollment_seal)]},
                {"phase": f"new_{count}:truth_free_apply", "dynamic_hash_inputs": [str(apply_seal)]},
                {"phase": f"new_{count}:isolated_score", "dynamic_result_inputs": [str(predictions / "predictor_result.json")]},
            ]
        )
    return plan


def _run(command: Sequence[str], *, log_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(value) for value in command],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return _parse_last_json_document(result.stdout)


def _parse_last_json_document(stdout: str) -> dict[str, Any]:
    """Parse the final complete JSON document from mixed or pretty stdout."""

    decoder = json.JSONDecoder()
    starts = [index for index, value in enumerate(stdout) if value in "[{"]
    for start in reversed(starts):
        try:
            value, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            continue
        if stdout[start + end :].strip():
            continue
        if not isinstance(value, dict):
            raise ValueError("launcher child result must be a JSON object")
        return value
    if stdout.strip():
        raise ValueError("launcher child completed without a final JSON object")
    return {"status": "PASS", "stdout_empty": True}


def _prepare_run_root(run_root: Path, *, resume: bool) -> bool:
    """Create a new run root or validate a cache-only interrupted run."""

    if not run_root.exists():
        if resume:
            raise FileNotFoundError("resume requested but JG020 run root does not exist")
        run_root.mkdir(parents=True, exist_ok=False)
        return False
    if not resume:
        raise FileExistsError("JG020 run root exists; explicit --resume is required")
    if (run_root / "execution_summary.json").exists():
        raise FileExistsError("completed JG020 run cannot be resumed")
    cache_manifest = run_root / "phase1_cache/cache_set.json"
    if not cache_manifest.is_file():
        raise FileNotFoundError("resume requires the completed Phase1 cache manifest")
    for count in (5, 10, 20):
        if (run_root / f"new_{count}").exists():
            raise FileExistsError("resume refuses a partially materialised Stage2-C row")
    return True


def execute(
    run_root: Path,
    python: str,
    *,
    resume: bool = False,
    reuse_cache_set: Path | None = None,
) -> dict[str, Any]:
    if run_root not in {DEFAULT_RUN_ROOT, RETRY1_RUN_ROOT, RETRY2_RUN_ROOT}:
        raise ValueError("JG020 run root is outside the locked default/retry roots")
    if reuse_cache_set is not None and resume:
        raise ValueError("external cache reuse and in-place resume are mutually exclusive")
    if reuse_cache_set is not None:
        resolved_cache = reuse_cache_set.resolve(strict=True)
        if resolved_cache != ORIGINAL_CACHE_SET:
            raise ValueError("retry1 may reuse only the original sealed JG020 cache set")
        _prepare_run_root(run_root, resume=False)
        cache_reused = True
        cache_manifest = resolved_cache
    else:
        cache_reused = _prepare_run_root(run_root, resume=bool(resume))
        cache_manifest = run_root / "phase1_cache/cache_set.json"
    logs = run_root / "logs"
    for required in (CHECKPOINT, GROUND_P4, MAPPING, CACHE_SPEC, SOURCE_ADAPTER, SOURCE_HEAD, VIEW_POLICY):
        if not required.is_file():
            raise FileNotFoundError(required)
    if sha256_file(CHECKPOINT) != "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98":
        raise ValueError("ADV3B02 checkpoint hash drift")
    if sha256_file(GROUND_P4) != "95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446":
        raise ValueError("ground P4 hash drift")
    if sha256_file(MAPPING) != "97af6115b51a6a3252e22315e40183c4c3efd7ccfeb1f16a61710028f72fda7f":
        raise ValueError("direct class mapping hash drift")
    if cache_reused:
        cache_result = {
            "status": "REUSED_EXISTING_SEALED_CACHE",
            "cache_set_manifest": str(cache_manifest),
            "cache_set_manifest_sha256": sha256_file(cache_manifest),
        }
    else:
        cache_result = _run(
            [python, "code/scripts/build_cvs_leo_weak_iq_cache.py", "--spec", str(CACHE_SPEC), "--device", "cuda:0"],
            log_path=logs / "phase1_offline_cache.log",
        )
    rows: list[dict[str, Any]] = []
    for count in (5, 10, 20):
        row = run_root / f"new_{count}"
        lock = REPO / f"paper_reproduction/configs/jg020_stage2c_candidate_lock_new{count}_20260716.json"
        validate_locked_candidate(json.loads(lock.read_text(encoding="utf-8")))
        source, source_scorer = row / "source_package", row / "source_scorer"
        source_seal = row / "source_package.seal.json"
        enrollment, enrollment_seal = row / "enrollment_package", row / "enrollment_package.seal.json"
        state = row / "enrollment_state"
        apply_root, apply_seal = row / "apply_package", row / "apply_package.seal.json"
        scorer, output, scores = row / "scorer", row / "predictor_output", row / "scores"
        source_result = _run(
            [
                python, "code/scripts/build_cvs_stage2_predictor_bundle.py",
                "--target-cache-set", str(cache_manifest),
                "--expected-cache-scope", "stage2_registered", "--predictor-out-root", str(source),
                "--scorer-out-root", str(source_scorer), "--detached-seal-path", str(source_seal),
                "--stage", "stage2c", "--receiver", "20-1", "--seed", "713101",
                "--old-class-labels", ",".join(OLD_LABELS), "--new-class-labels", ",".join(NEW_LABELS[:count]),
                "--new-class-count", str(count), "--support-pool-max-k", "10", "--query-per-tx", "20",
                "--candidate-lock", str(lock), "--checkpoint", str(CHECKPOINT), "--adapter", str(SOURCE_ADAPTER),
                "--head-artifact", str(SOURCE_HEAD), "--tta-policy-json", str(VIEW_POLICY),
            ],
            log_path=logs / f"new_{count}_source_bundle.log",
        )
        source_seal_sha = sha256_file(source_seal)
        enrollment_result = _run(
            [
                python, "paper_reproduction/scripts/build_cvs_jg020_split_packages.py", "enrollment",
                "--source-package-root", str(source), "--source-detached-seal", str(source_seal),
                "--source-expected-seal-sha256", source_seal_sha, "--candidate-lock", str(lock),
                "--output-root", str(enrollment), "--output-detached-seal", str(enrollment_seal),
                "--checkpoint-full", str(CHECKPOINT), "--ground-adapter", str(GROUND_P4),
                "--direct-class-mapping", str(MAPPING),
            ],
            log_path=logs / f"new_{count}_split_enrollment.log",
        )
        enrollment_seal_sha = sha256_file(enrollment_seal)
        enroll_result = _run(
            [
                python, "paper_reproduction/scripts/enroll_cvs_jg020_support_only.py",
                "--package-root", str(enrollment), "--detached-seal", str(enrollment_seal),
                "--expected-seal-sha256", enrollment_seal_sha, "--output-root", str(state),
                "--device", "cuda:0", "--batch-size", "256",
            ],
            log_path=logs / f"new_{count}_enroll.log",
        )
        apply_result = _run(
            [
                python, "paper_reproduction/scripts/build_cvs_jg020_split_packages.py", "apply",
                "--source-package-root", str(source), "--source-detached-seal", str(source_seal),
                "--source-expected-seal-sha256", source_seal_sha, "--candidate-lock", str(lock),
                "--output-root", str(apply_root), "--output-detached-seal", str(apply_seal),
                "--enrollment-package-root", str(enrollment), "--enrollment-detached-seal", str(enrollment_seal),
                "--enrollment-expected-seal-sha256", enrollment_seal_sha,
                "--candidate-runtime", str(state / "candidate_runtime.ts"),
                "--identity-runtime", str(state / "identity_runtime.ts"),
                "--direct-runtime", str(state / "direct_runtime.ts"),
                "--prototype-head", str(state / "prototype_head.npz"),
                "--enrollment-receipt", str(state / "enrollment_receipt.json"),
                "--source-scoring-manifest", str(source_scorer / "scoring_manifest.json"),
                "--scorer-out-root", str(scorer),
            ],
            log_path=logs / f"new_{count}_split_apply.log",
        )
        apply_seal_sha = sha256_file(apply_seal)
        predict_result = _run(
            [
                python, "paper_reproduction/scripts/run_cvs_jg020_apply_only_predictor.py",
                "--package-root", str(apply_root), "--detached-seal", str(apply_seal),
                "--expected-seal-sha256", apply_seal_sha, "--output-dir", str(output),
                "--device", "cuda:0", "--batch-size", "256",
            ],
            log_path=logs / f"new_{count}_predict.log",
        )
        output.mkdir(parents=True, exist_ok=True)
        _write_json_new(output / "predictor_result.json", predict_result)
        scores.mkdir(parents=True, exist_ok=False)
        score_result = _run(
            [
                python, "code/scripts/score_cvs_stage2_sealed_prediction.py",
                "--prediction-artifact", predict_result["prediction_artifact"],
                "--expected-prediction-artifact-sha256", predict_result["prediction_artifact_sha256"],
                "--expected-prediction-seal-sha256", predict_result["prediction_seal_sha256"],
                "--scoring-manifest", str(scorer / "scoring_manifest.json"),
                "--expected-scoring-manifest-sha256", sha256_file(scorer / "scoring_manifest.json"),
                "--formal-rows", str(scores / "formal_rows.json"),
                "--formal-predictions", str(scores / "formal_predictions.json"),
                "--scoring-receipt", str(scores / "scoring_receipt.json"),
            ],
            log_path=logs / f"new_{count}_score.log",
        )
        rows.append(
            {
                "new_class_count": count,
                "source": source_result,
                "enrollment_package": enrollment_result,
                "enrollment": enroll_result,
                "apply_package": apply_result,
                "prediction": predict_result,
                "score": score_result,
                "experimental_adapter_fit_count": 1,
            }
        )
    summary = {
        "schema": "cvs.jg020.stage2c.dev_execution_summary.v1",
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "cache": cache_result,
        "rows": rows,
        "experimental_repeated_adapter_fit_count": 3,
        "deployment_amortized_adapter_fit_count": 1,
        "deployment_registration_sequence": [5, 10, 20],
        "deployment_new_support_gradient_used": False,
        "deployment_registration_operation": "frozen_runtime_support_forward_plus_prototype_append",
    }
    _write_json_new(run_root / "execution_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-cache-set", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan-out", type=Path)
    args = parser.parse_args()
    plan = {
        "schema": "cvs.jg020.stage2c.dev_command_plan.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_root": str(args.run_root),
        "experimental_repeated_adapter_fit_count": 3,
        "deployment_amortized_adapter_fit_count": 1,
        "resume": bool(args.resume),
        "reuse_cache_set": str(args.reuse_cache_set) if args.reuse_cache_set else None,
        "commands": _command_plan(args.run_root, args.python, cache_set=args.reuse_cache_set),
    }
    if not args.execute:
        if args.plan_out:
            _write_json_new(args.plan_out, plan)
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0
    print(
        json.dumps(
            execute(
                args.run_root,
                args.python,
                resume=bool(args.resume),
                reuse_cache_set=args.reuse_cache_set,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
