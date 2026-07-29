"""Build the 99 missing Stage2 feature-cache triples for the fresh v9 input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from package_completion_controller import (
    _normalize_exception_fingerprint,
    _run_in_waves,
)


RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
METHOD_SEEDS = (7283101, 7283102, 7283103, 7283104, 7283105)
K_SHOTS = (1, 2, 5, 10)
GPU_SLOTS = (1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)
REUSED_RECEIVER = "20-1"
REUSED_METHOD_SEED = 7283101
REUSED_K_SHOT = 10
NEW_CLASS_DRAW_SEED = 7282401
EXPECTED_TASKS = 99
SCOPES_PER_CALL = 3
FILES_PER_SCOPE = 2
REUSED_SCOPE_CACHES = 3
EXPECTED_TOTAL_SCOPE_CACHES = 300
CANONICAL_STAGE2A_TARGET = 25
STAGE2B_COMBO_TARGET = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    return parser.parse_args()


def _package_base(args: argparse.Namespace, receiver: str, method_seed: int) -> Path:
    if receiver == REUSED_RECEIVER and method_seed == REUSED_METHOD_SEED:
        return args.reuse_root
    return (
        args.input_root
        / "artifacts"
        / "packages"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{method_seed}"
    )


def _feature_root(
    input_root: Path, receiver: str, method_seed: int, k_shot: int
) -> Path:
    return (
        input_root
        / "artifacts"
        / "features"
        / f"rx_{receiver.replace('-', '_')}"
        / f"method_{method_seed}"
        / f"k_{k_shot}"
    )


def _scope_counts(succeeded_calls: int) -> dict[str, int]:
    return {
        "generated_scope_caches": succeeded_calls * SCOPES_PER_CALL,
        "generated_physical_files": (
            succeeded_calls * SCOPES_PER_CALL * FILES_PER_SCOPE
        ),
        "reused_scope_caches": REUSED_SCOPE_CACHES,
        "expected_total_scope_caches": EXPECTED_TOTAL_SCOPE_CACHES,
        "canonical_stage2a_target": CANONICAL_STAGE2A_TARGET,
        "stage2b_combo_target": STAGE2B_COMBO_TARGET,
    }


def _build_command(
    args: argparse.Namespace,
    *,
    receiver: str,
    method_seed: int,
    k_shot: int,
) -> list[str]:
    package = _package_base(args, receiver, method_seed)
    before_seal = package / "before" / "predictor.seal.json"
    after_seal = package / "new20" / "predictor.seal.json"
    support_seed = method_seed + 100
    query_seed = method_seed + 200
    cache_seed = method_seed - 6_570_000
    return [
        "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python",
        str(
            args.release_root
            / "code"
            / "scripts"
            / "build_full_ablation_stage2_feature_cache.py"
        ),
        "--before-package-root",
        str(package / "before" / "predictor"),
        "--before-seal-path",
        str(before_seal),
        "--before-seal-sha256",
        _sha256(before_seal),
        "--after-package-root",
        str(package / "new20" / "predictor"),
        "--after-seal-path",
        str(after_seal),
        "--after-seal-sha256",
        _sha256(after_seal),
        "--phase1-deployment-binding-path",
        str(args.phase1_root / "artifacts/phase1_final/deployment_binding.json"),
        "--ground-component-dir",
        str(args.phase1_root / "artifacts/phase1_unsigned/package/component"),
        "--ground-manifest-sha256",
        "03b5761d9cfd0f09a6b64710f5ebe7c270314bf5d73215206e5e8cf84606448a",
        "--phase1-prototype-path",
        str(
            args.phase1_root
            / "artifacts/phase1_normalized/deployment_prototype"
            / "phase2_zid_prototypes.pt"
        ),
        "--phase1-prototype-manifest-path",
        str(
            args.phase1_root
            / "artifacts/phase1_normalized/deployment_prototype"
            / "phase2_zid_prototypes.json"
        ),
        "--expected-phase1-prototype-sha256",
        "e0e10b671dec5088bcb6e59b475dc3a99060b0ccbc03581e345a5e953b6088f0",
        "--expected-phase1-prototype-manifest-sha256",
        "89c1f21a5476e8d6b6a27264af6505d9ac4ab6eb66b32ea1e54c1d21405fc527",
        "--expected-phase1-bundle-sha256",
        "1eb6d07b9d6339400892c5553f33261f40513922d4b08c907446e44e993307d7",
        "--cache-output-root",
        str(_feature_root(args.input_root, receiver, method_seed, k_shot)),
        "--phase2-data-status",
        "VALIDATED_ONCE",
        "--capsule-id",
        (
            f"d18-rx{receiver}-cache{cache_seed}-m{method_seed}"
            f"-k{k_shot}-new20"
        ),
        "--split-id",
        (
            f"p2_min_v1-rx{receiver}-m{method_seed}-s{support_seed}"
            f"-q{query_seed}-d{NEW_CLASS_DRAW_SEED}-k{k_shot}"
        ),
        "--k-shot",
        str(k_shot),
        "--method-seed",
        str(method_seed),
        "--support-seed",
        str(support_seed),
        "--query-seed",
        str(query_seed),
        "--new-class-draw-seed",
        str(NEW_CLASS_DRAW_SEED),
        "--device",
        "cuda:0",
    ]


def main() -> int:
    args = _parse_args()
    feature_base = args.input_root / "artifacts" / "features"
    log_root = args.input_root / "logs" / "feature_completion"
    if feature_base.exists() or log_root.exists():
        raise FileExistsError("fresh feature/log roots must be absent")
    tasks: list[dict[str, object]] = []
    slot_index = 0
    for receiver in RECEIVERS:
        for method_seed in METHOD_SEEDS:
            for k_shot in K_SHOTS:
                if (
                    receiver == REUSED_RECEIVER
                    and method_seed == REUSED_METHOD_SEED
                    and k_shot == REUSED_K_SHOT
                ):
                    continue
                command = _build_command(
                    args,
                    receiver=receiver,
                    method_seed=method_seed,
                    k_shot=k_shot,
                )
                tasks.append(
                    {
                        "receiver": receiver,
                        "method_seed": method_seed,
                        "k_shot": k_shot,
                        "gpu": GPU_SLOTS[slot_index % len(GPU_SLOTS)],
                        "command": command,
                    }
                )
                slot_index += 1
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, got {len(tasks)}")

    log_root.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def run_one(task: dict[str, object]) -> dict[str, object]:
        key = (
            f"rx_{str(task['receiver']).replace('-', '_')}"
            f"_m{task['method_seed']}_k{task['k_shot']}"
        )
        log_path = log_root / f"{key}.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
        with log_path.open("xb") as stream:
            process = subprocess.run(
                list(task["command"]),
                cwd=args.release_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        fingerprint = (
            _normalize_exception_fingerprint(log_path)
            if process.returncode != 0
            else ""
        )
        return {
            "key": key,
            "receiver": task["receiver"],
            "method_seed": task["method_seed"],
            "k_shot": task["k_shot"],
            "gpu": task["gpu"],
            "returncode": process.returncode,
            "exception_fingerprint": fingerprint,
            "log": str(log_path),
            "output": str(
                _feature_root(
                    args.input_root,
                    str(task["receiver"]),
                    int(task["method_seed"]),
                    int(task["k_shot"]),
                )
            ),
        }

    results, exception_fingerprints, systemic_stop = _run_in_waves(
        tasks,
        max_workers=len(GPU_SLOTS),
        runner=run_one,
    )
    for result in results:
        print(json.dumps(result, sort_keys=True), flush=True)
    succeeded = sum(row["returncode"] == 0 for row in results)
    summary = {
        "schema": "cvs.stage2.v9.feature_completion.v1",
        "expected_build": EXPECTED_TASKS,
        "reused_feature_triples": 1,
        "launched": len(results),
        "completed": len(results),
        "succeeded": succeeded,
        "failed": sum(row["returncode"] != 0 for row in results),
        "not_launched": EXPECTED_TASKS - len(results),
        "exception_fingerprints": exception_fingerprints,
        "systemic_stop": systemic_stop,
        "gpu_slots": list(GPU_SLOTS),
        "elapsed_sec": time.time() - started,
        "results": results,
        **_scope_counts(succeeded),
    }
    output = args.input_root / "source" / "feature_completion_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "launched",
                    "completed",
                    "succeeded",
                    "failed",
                    "not_launched",
                    "exception_fingerprints",
                    "systemic_stop",
                    "generated_scope_caches",
                    "generated_physical_files",
                    "reused_scope_caches",
                    "expected_total_scope_caches",
                    "canonical_stage2a_target",
                    "stage2b_combo_target",
                    "elapsed_sec",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
