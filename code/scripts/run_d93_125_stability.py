#!/usr/bin/env python3
"""Run a locked D93 paired-ground transport 125-job stability screen."""

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
from cvsrffi.stage2_d93_query_evaluation import CANDIDATES_D93


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
        raise base.StabilityLauncherError("D93 cpu thread count must be positive")
    configured = {name: str(threads) for name in _CPU_THREAD_ENV_VARS}
    configured["CVSRFFI_CPU_THREADS"] = str(threads)
    configured["CVSRFFI_CPU_INTEROP_THREADS"] = "1"
    os.environ.update(configured)
    return configured


def _d93_job_command(
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
    result.add_argument("--candidate", choices=CANDIDATES_D93, required=True)
    result.add_argument("--ground-component-dir", required=True)
    result.add_argument("--ground-manifest-sha256", required=True)
    result.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="CPU threads per D93 row for feature and covariance work",
    )
    return result


def main() -> int:
    global _GROUND_COMPONENT_DIR, _GROUND_MANIFEST_SHA256
    args = parser().parse_args()
    _GROUND_COMPONENT_DIR = str(args.ground_component_dir)
    _GROUND_MANIFEST_SHA256 = str(args.ground_manifest_sha256).lower()
    if len(_GROUND_MANIFEST_SHA256) != 64:
        raise base.StabilityLauncherError("D93 ground manifest SHA drift")
    cpu_thread_env = _configure_child_cpu_threads(args.cpu_threads)
    base.CANDIDATE = str(args.candidate)
    base.ROW_PIPELINE = CODE_ROOT / "scripts" / "run_cvs_somph_diag_row_pipeline.py"
    if not hasattr(base, "_ORIGINAL_JOB_COMMAND"):
        base._ORIGINAL_JOB_COMMAND = base._job_command
    base._job_command = _d93_job_command
    result = base.run(args)
    result["cpu_thread_env"] = cpu_thread_env
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] in {"PASS", "MANIFEST_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
