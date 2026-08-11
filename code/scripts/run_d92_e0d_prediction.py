#!/usr/bin/env python3
"""Run one truth-free D92-E0D prediction job from sealed predictor packages."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_e0d_query_evaluation import (  # noqa: E402
    run_d92_e0d_query_evaluation,
)
from cvsrffi.stage2_d92_e0d_slim import D92_E0D_ARMS  # noqa: E402


class D92E0DPredictionEntryError(RuntimeError):
    """Raised when an immutable E0D prediction boundary would be violated."""


def _readonly_file(path: Path) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise D92E0DPredictionEntryError(
            f"prediction output is not read-only: {path}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root)
    if output.exists():
        raise D92E0DPredictionEntryError(
            f"prediction output already exists; refusing overwrite: {output}"
        )
    result = run_d92_e0d_query_evaluation(
        arm_id=args.arm,
        before_enrollment_package_root=args.before_enrollment_package_root,
        before_enrollment_seal_path=args.before_enrollment_seal_path,
        before_enrollment_seal_sha256=args.before_enrollment_seal_sha256,
        before_apply_package_root=args.before_apply_package_root,
        before_apply_seal_path=args.before_apply_seal_path,
        before_apply_seal_sha256=args.before_apply_seal_sha256,
        after_enrollment_package_root=args.after_enrollment_package_root,
        after_enrollment_seal_path=args.after_enrollment_seal_path,
        after_enrollment_seal_sha256=args.after_enrollment_seal_sha256,
        after_apply_package_root=args.after_apply_package_root,
        after_apply_seal_path=args.after_apply_seal_path,
        after_apply_seal_sha256=args.after_apply_seal_sha256,
        ground_component_dir=args.ground_component_dir,
        ground_manifest_sha256=args.ground_manifest_sha256,
        output_root=output,
        device=args.device,
    )
    for state in ("before", "after"):
        _readonly_file(output / state / "prediction_artifact.npz")
        _readonly_file(output / state / "COMMIT.json")
    return {
        "schema": "cvs.phase2.d92_e0d.truth_free_prediction_entry.v1",
        "status": "D92_E0D_TRUTH_FREE_PREDICTIONS_COMPLETE",
        "candidate": result["candidate"],
        "arm_id": result["arm_id"],
        "output_root": str(output.resolve()),
        "query_truth_access": False,
        "query_fit_access": False,
        "query_update_access": False,
        "query_selection_access": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--before-enrollment-package-root", required=True)
    result.add_argument("--before-enrollment-seal-path", required=True)
    result.add_argument("--before-enrollment-seal-sha256", required=True)
    result.add_argument("--before-apply-package-root", required=True)
    result.add_argument("--before-apply-seal-path", required=True)
    result.add_argument("--before-apply-seal-sha256", required=True)
    result.add_argument("--after-enrollment-package-root", required=True)
    result.add_argument("--after-enrollment-seal-path", required=True)
    result.add_argument("--after-enrollment-seal-sha256", required=True)
    result.add_argument("--after-apply-package-root", required=True)
    result.add_argument("--after-apply-seal-path", required=True)
    result.add_argument("--after-apply-seal-sha256", required=True)
    result.add_argument("--ground-component-dir", required=True)
    result.add_argument("--ground-manifest-sha256", required=True)
    result.add_argument("--arm", choices=tuple(D92_E0D_ARMS), required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--device", required=True)
    return result


def main() -> int:
    receipt = run(parser().parse_args())
    print(json.dumps(receipt, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
