#!/usr/bin/env python3
"""D91 support-only crossfit-consensus ground sigma diagnostic probe."""

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
CORE_PATH = CODE_ROOT / "cvsrffi" / "stage2_d91_crossfit_consensus_sigma_margin.py"

ARM = "crossfit_consensus_ground_sigma_margin_head"
FORMULA = (
    "reuse the exact D87 immutable compressed-v2 ground radius sigma geometry "
    "and 20-step support-only head fit; compute the initial sigma-risk gradient "
    "independently in each physical-rank OOF fold, normalize every fold "
    "gradient, multiply the D87 centered residual by the clipped mean "
    "off-diagonal cosine agreement without a threshold, and compile the "
    "result into the unchanged single INT8 affine head"
)


class D91ProbeError(RuntimeError):
    """Raised when D91 integration or evidence closure drifts."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise D91ProbeError(f"D91 could not load {path}")
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


d87 = _load("d91_d87_probe_scaffold", D87_PROBE_PATH)
core = _load("d91_crossfit_consensus_core", CORE_PATH)
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
    shots = int(k_shot)
    classes = int(class_count)
    rank = 13
    domains = 14
    per_fold_sigma_gradient = classes * classes * rank * (2 + 2 * domains)
    agreement = shots * shots * classes * rank
    extra = int(shots * per_fold_sigma_gradient + agreement)
    repeated_lda = int(lda_macs)
    inherited["d91_consensus_crossfit_lda_fit_count"] = shots
    inherited["d91_consensus_crossfit_lda_fit_macs"] = repeated_lda
    inherited["d91_fold_consensus_mac_upper_bound"] = extra
    inherited["frank_wolfe_mac_upper_bound"] += extra
    inherited["non_lda_total"] += extra
    inherited["crossfit_lda_fit_macs"] += repeated_lda
    inherited["total_added"] += repeated_lda + extra
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
        row
        for row in rows
        if row.get("candidate_id")
        in ("D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED")
    ]
    factors: list[float] = []
    active = 0
    for row in target:
        audit = row["geometry_summary"]["d79_worstclass_margin_audit"]
        resource = row["resource"]
        factor = float(audit["consensus_factor"])
        if (
            audit["schema"]
            != "cvs.phase2.d91.crossfit_consensus_sigma_margin_audit.v1"
            or not 0.0 <= factor <= 1.0
            or int(audit["fold_gradient_count"]) != 8
            or audit["class_permutation_equivariant"] is not True
            or audit["old_new_role_specific_branch"] is not False
            or audit["class_id_specific_formula"] is not False
            or audit["physical_group_crossfit_preserved"] is not True
            or int(audit["query_rows_used"]) != 0
            or float(audit["residual_logit_at_support_center_max_abs"]) > 1.0e-5
            or audit["residual_sha256"]
            == audit["d87_unshrunk_audit"]["residual_sha256"]
            and factor not in (0.0, 1.0)
            or int(resource["descendant_extra_crossfit_lda_fit_count"]) != 8
            or int(resource["descendant_actual_crossfit_lda_fit_count"]) != 16
            or int(resource["descendant_extra_support_mac_upper_bound"]) != 386_672
        ):
            raise D91ProbeError("D91 consensus audit closure drift")
        factors.append(factor)
        active += int(audit["residual_active"])
    return {
        **evidence,
        "verified_d91_target_row_count": len(target),
        "verified_d91_active_count": active,
        "verified_d91_consensus_factor_min": min(factors),
        "verified_d91_consensus_factor_mean": sum(factors) / len(factors),
        "verified_d91_consensus_factor_max": max(factors),
        "verified_d91_query_rows_used": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d91-arm", required=True, choices=(ARM,))
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
    d87.EXTRA_CROSSFIT_LDA_FIT_COUNT = 8
    d87.EXTRA_SUPPORT_MAC_UPPER_BOUND = 386_672
    translated = ["--d87-arm", ARM, *runner_arguments]
    exit_code = int(d87.main(translated))
    if exit_code != 0:
        return exit_code
    inherited_path = output / "D87_PROBE_METADATA.json"
    metadata = json.loads(inherited_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "schema": "cvs.phase2.d91.crossfit_consensus_sigma_probe.v1",
            "arm": ARM,
            "formula": FORMULA,
            "d91_core_sha256": _sha256(CORE_PATH),
            "d91_probe_sha256": _sha256(Path(__file__).resolve()),
            "inherited_d87_scaffold_metadata_sha256": _sha256(inherited_path),
            "forced_nonpromotable": True,
        }
    )
    (output / "D91_PROBE_METADATA.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
