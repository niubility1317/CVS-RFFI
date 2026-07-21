#!/usr/bin/env python3
"""Run the locked D92 role-only Oracle licensed upper-bound 125 screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts import run_cvs_somph_diag_125_stability as base
from cvsrffi.stage2_d92_role_oracle_query_evaluation import (
    CANDIDATE_D92_ROLE_ORACLE,
    LICENSE_STATUS,
)


_GROUND_COMPONENT_DIR = ""
_GROUND_MANIFEST_SHA256 = ""
_CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _configure_child_cpu_threads(cpu_threads: int) -> dict[str, str]:
    threads = int(cpu_threads)
    if threads <= 0:
        raise base.StabilityLauncherError(
            "D92 Role-Oracle cpu thread count must be positive"
        )
    configured = {name: str(threads) for name in _CPU_THREAD_ENV_VARS}
    configured["CVSRFFI_CPU_THREADS"] = str(threads)
    configured["CVSRFFI_CPU_INTEROP_THREADS"] = "1"
    os.environ.update(configured)
    return configured


def _job_command(
    job: Mapping[str, Any],
    *,
    phase1_checkpoint: str,
    sealed_runtime: str,
    method_lock: str,
    device: str,
) -> list[str]:
    command = base._ORIGINAL_JOB_COMMAND(
        job,
        phase1_checkpoint=phase1_checkpoint,
        sealed_runtime=sealed_runtime,
        method_lock=method_lock,
        device=device,
    )
    command.extend(
        [
            "--ground-component-dir",
            _GROUND_COMPONENT_DIR,
            "--ground-manifest-sha256",
            _GROUND_MANIFEST_SHA256,
        ]
    )
    return command


def parser() -> argparse.ArgumentParser:
    result = base.parser()
    result.description = __doc__
    result.add_argument("--ground-component-dir", required=True)
    result.add_argument("--ground-manifest-sha256", required=True)
    result.add_argument("--cpu-threads", type=int, default=2)
    return result


def main() -> int:
    global _GROUND_COMPONENT_DIR, _GROUND_MANIFEST_SHA256
    args = parser().parse_args()
    _GROUND_COMPONENT_DIR = str(args.ground_component_dir)
    _GROUND_MANIFEST_SHA256 = str(args.ground_manifest_sha256).lower()
    if len(_GROUND_MANIFEST_SHA256) != 64 or any(
        value not in "0123456789abcdef" for value in _GROUND_MANIFEST_SHA256
    ):
        raise base.StabilityLauncherError(
            "D92 Role-Oracle ground manifest SHA drift"
        )
    cpu_thread_env = _configure_child_cpu_threads(args.cpu_threads)
    ground_root = str(Path(_GROUND_COMPONENT_DIR).resolve(strict=True))
    original = {
        "candidate": base.CANDIDATE,
        "claim_scope": base.CLAIM_SCOPE,
        "formal_launch_authority": base.FORMAL_LAUNCH_AUTHORITY,
        "phase2_contract": base.PHASE2_CONTRACT,
        "row_pipeline": base.ROW_PIPELINE,
        "job_command": base._job_command,
        "build_manifest": base.build_manifest,
        "had_original_job_command": hasattr(base, "_ORIGINAL_JOB_COMMAND"),
        "original_job_command_attr": getattr(base, "_ORIGINAL_JOB_COMMAND", None),
    }
    licensed_contract = {
        **dict(original["phase2_contract"]),
        "phase2_query_decision_policy": (
            "per_sample_role_partition_all_registered_classes_within_role"
        ),
        "phase2_query_role_oracle_access": True,
        "licensed_protocol_deviation": "query_old_new_role_oracle_only",
        "formal_protocol_valid": False,
        "promotion_eligible": False,
    }
    original_build_manifest = base.build_manifest

    def build_manifest_with_ground(**kwargs: Any) -> dict[str, Any]:
        manifest = original_build_manifest(**kwargs)
        manifest.update(
            {
                "ground_component_dir": ground_root,
                "ground_manifest_sha256": _GROUND_MANIFEST_SHA256,
                "cpu_threads_per_row": int(args.cpu_threads),
                "result_label": LICENSE_STATUS,
                "formal_protocol_valid": False,
                "promotion_eligible": False,
            }
        )
        for job in manifest["jobs"]:
            job["ground_component_dir"] = ground_root
            job["ground_manifest_sha256"] = _GROUND_MANIFEST_SHA256
        return manifest

    try:
        base.CANDIDATE = CANDIDATE_D92_ROLE_ORACLE
        base.CLAIM_SCOPE = LICENSE_STATUS
        base.FORMAL_LAUNCH_AUTHORITY = False
        base.PHASE2_CONTRACT = licensed_contract
        base.ROW_PIPELINE = (
            CODE_ROOT / "scripts" / "run_cvs_somph_diag_row_pipeline.py"
        )
        base._ORIGINAL_JOB_COMMAND = original["job_command"]
        base._job_command = _job_command
        base.build_manifest = build_manifest_with_ground
        result = base.run(args)
    finally:
        base.CANDIDATE = original["candidate"]
        base.CLAIM_SCOPE = original["claim_scope"]
        base.FORMAL_LAUNCH_AUTHORITY = original["formal_launch_authority"]
        base.PHASE2_CONTRACT = original["phase2_contract"]
        base.ROW_PIPELINE = original["row_pipeline"]
        base._job_command = original["job_command"]
        base.build_manifest = original["build_manifest"]
        if original["had_original_job_command"]:
            base._ORIGINAL_JOB_COMMAND = original["original_job_command_attr"]
        else:
            delattr(base, "_ORIGINAL_JOB_COMMAND")
    result.update(
        {
            "result_label": LICENSE_STATUS,
            "formal_protocol_valid": False,
            "promotion_eligible": False,
            "cpu_thread_env": cpu_thread_env,
        }
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] in {"PASS", "MANIFEST_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
