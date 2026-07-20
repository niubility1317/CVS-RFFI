#!/usr/bin/env python3
"""D88 support-only ground sigma Pareto-guard diagnostic probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
D87_PROBE_PATH = SCRIPT_DIR / "probe_d87_ground_radius_sigma_margin.py"
CORE_PATH = CODE_ROOT / "cvsrffi" / "stage2_d88_ground_sigma_pareto_guard.py"

ARM = "ground_sigma_pareto_guard_centered_head"
FORMULA = (
    "reuse the exact D87 immutable v2 radius sigma geometry and fixed "
    "smooth-worst objective; at each of 20 support-only steps project the "
    "sigma descent direction onto every registered class clean-OOF CE "
    "descent halfspace with the same label-permutation-equivariant formula; "
    "accept only an objective-nonincreasing step whose exact clean OOF CE "
    "does not increase for any class; compile the centered residual into "
    "the unchanged single INT8 affine head"
)


class D88ProbeError(RuntimeError):
    """Raised when the D88 integration or evidence closure drifts."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise D88ProbeError(f"D88 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


d87 = _load("d88_d87_probe_scaffold", D87_PROBE_PATH)
core = _load("d88_ground_sigma_pareto_core", CORE_PATH)
_D87_VERIFY = d87._verify_output
_D87_RESOURCE_UPPER_BOUNDS = d87._resource_upper_bounds


def _resource_upper_bounds(
    *,
    k_shot: int,
    class_count: int,
    dimension: int,
    lda_macs: int,
    ground_statistics_macs: int,
) -> dict[str, int]:
    inherited = _D87_RESOURCE_UPPER_BOUNDS(
        k_shot=k_shot,
        class_count=class_count,
        dimension=dimension,
        lda_macs=lda_macs,
        ground_statistics_macs=ground_statistics_macs,
    )
    classes = int(class_count)
    held = int(k_shot) * classes
    rank = 13
    clean_gradient_macs = int(
        core.OPTIMIZER_STEPS * classes * held * classes * (2 * rank + 4)
    )
    cone_projection_macs = int(
        core.OPTIMIZER_STEPS
        * core.CONE_PROJECTION_SWEEPS
        * classes
        * classes
        * rank
        * 8
    )
    pareto_macs = clean_gradient_macs + cone_projection_macs
    inherited["d88_clean_class_gradient_mac_upper_bound"] = clean_gradient_macs
    inherited["d88_cone_projection_mac_upper_bound"] = cone_projection_macs
    inherited["d88_pareto_guard_mac_upper_bound"] = pareto_macs
    inherited["frank_wolfe_mac_upper_bound"] += pareto_macs
    inherited["non_lda_total"] += pareto_macs
    inherited["total_added"] += pareto_macs
    return inherited


def _verify_output(
    output: Path, script_sha: str, helper_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence = _D87_VERIFY(output, script_sha, helper_hashes)
    rows = [
        json.loads(line)
        for line in (output / "training_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    target = [
        row for row in rows
        if row.get("candidate_id") in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    active = 0
    max_class_delta: list[float] = []
    projection_counts: list[int] = []
    for row in target:
        audit = row["geometry_summary"]["d79_worstclass_margin_audit"]
        if (
            audit["schema"] != "cvs.phase2.d88.ground_sigma_pareto_audit.v1"
            or audit["all_class_clean_ce_nonincrease_verified"] is not True
            or float(audit["oof_clean_ce_delta_max_class"])
            > float(audit["clean_pareto_guard_tolerance"]) + 1.0e-12
            or audit["class_permutation_equivariant"] is not True
            or audit["old_new_role_specific_branch"] is not False
            or audit["class_id_specific_formula"] is not False
            or int(audit["query_rows_used"]) != 0
        ):
            raise D88ProbeError("D88 Pareto audit closure drift")
        if any(
            float(item["clean_ce_max_class_delta_step"])
            > 1.0e-10
            for item in audit["optimizer_objective_trace"]
        ):
            raise D88ProbeError("D88 per-step class guard drift")
        active += int(audit["residual_active"])
        max_class_delta.append(float(audit["oof_clean_ce_delta_max_class"]))
        projection_counts.append(int(audit["total_halfspace_projection_count"]))
    return {
        **evidence,
        "verified_d88_target_row_count": len(target),
        "verified_d88_active_count": active,
        "verified_d88_max_class_clean_ce_delta_max": max(max_class_delta),
        "verified_d88_halfspace_projection_count_min": min(projection_counts),
        "verified_d88_halfspace_projection_count_max": max(projection_counts),
        "verified_d88_query_rows_used": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d88-arm", required=True, choices=(ARM,))
    return parser


def main(argv: list[str] | None = None) -> int:
    known, runner_arguments = build_parser().parse_known_args(argv)
    output = d87.d43._runner_output(runner_arguments)
    d87.ARM = ARM
    d87.FORMULA = FORMULA
    d87.CORE_PATH = CORE_PATH
    d87.core = core
    d87.__file__ = str(Path(__file__).resolve())
    d87._verify_output = _verify_output
    d87._resource_upper_bounds = _resource_upper_bounds

    original_fit = core.fit_ground_sigma_pareto_guard

    def fit_alias(*args: Any, **kwargs: Any):
        return original_fit(*args, **kwargs)

    core.fit_ground_radius_sigma_margin = fit_alias
    translated = [
        "--d87-arm", ARM,
        *runner_arguments,
    ]
    exit_code = int(d87.main(translated))
    if exit_code != 0:
        return exit_code
    inherited_path = output / "D87_PROBE_METADATA.json"
    metadata = json.loads(inherited_path.read_text(encoding="utf-8"))
    metadata.update({
        "schema": "cvs.phase2.d88.ground_sigma_pareto_probe.v1",
        "arm": ARM,
        "formula": FORMULA,
        "d88_core_sha256": _sha256(CORE_PATH),
        "d88_probe_sha256": _sha256(Path(__file__).resolve()),
        "inherited_d87_scaffold_metadata_sha256": _sha256(inherited_path),
        "forced_nonpromotable": True,
    })
    (output / "D88_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
