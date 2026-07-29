#!/usr/bin/env python3
"""Build fail-closed Phase1 or Phase2 full-ablation plans.

The generated plan is a preregistered identity surface, not launch authority.
No command in this file starts a training or evaluation process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cvsrffi.full_ablation_spec import (
    DESIGN_ID,
    PHASE1_T1_ARMS,
    PHASE2_T1_ARMS,
    ArmSpec,
    FullAblationSpecError,
    SeedBundle,
    build_phase1_label_rows,
    build_phase1_t1_rows,
    build_phase2_rows,
    validate_stage2_registry_disjointness,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_REGISTRY = (
    ROOT / "configs" / "full_ablation_20260728" / "seed_registry.json"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_text_sha256(payload: bytes) -> str:
    """Hash Git/Linux text bytes, independent of Windows checkout CRLF."""

    return _sha256_bytes(payload.replace(b"\r\n", b"\n"))


def _load_registry(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    registry = json.loads(payload.decode("utf-8-sig"))
    if registry.get("schema") != "cvs.full_ablation.seed_registry.v1":
        raise FullAblationSpecError("unexpected seed-registry schema")
    if registry.get("design_id") != DESIGN_ID:
        raise FullAblationSpecError("seed registry belongs to another design")
    screening = registry.get("stage2_screening") or {}
    confirmation = registry.get("stage2_confirmation") or {}
    validate_stage2_registry_disjointness(
        [
            SeedBundle(**item)
            for item in screening.get("seed_bundles") or []
        ],
        screening.get("new_class_draw_seeds") or [],
        [
            SeedBundle(**item)
            for item in confirmation.get("seed_bundles") or []
        ],
        confirmation.get("new_class_draw_seeds") or [],
    )
    return registry, _canonical_text_sha256(payload)


def _select_arms(raw_ids: str) -> list[ArmSpec]:
    available = {arm.ablation_id: arm for arm in PHASE2_T1_ARMS}
    if raw_ids.strip().lower() == "t1":
        return list(PHASE2_T1_ARMS)
    ids = [value.strip() for value in raw_ids.split(",") if value.strip()]
    if not ids:
        raise FullAblationSpecError("at least one Phase2 arm ID is required")
    unknown = [value for value in ids if value not in available]
    if unknown:
        raise FullAblationSpecError(
            "unknown Phase2 arm IDs: " + ",".join(unknown)
        )
    return [available[value] for value in ids]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    registry_path = Path(args.seed_registry).resolve()
    registry, registry_hash = _load_registry(registry_path)
    if args.phase == "phase1":
        registered_phase1_train_seeds = [
            int(value) for value in registry["phase1_train_seeds"]
        ]
        if args.phase1_matrix == "label":
            rows = build_phase1_label_rows(
                registered_phase1_train_seeds,
                git_commit=args.git_commit,
            )
            stage = "label"
        else:
            rows = build_phase1_t1_rows(
                registered_phase1_train_seeds,
                git_commit=args.git_commit,
            )
            stage = "t1"
    else:
        registered_phase1_train_seeds = []
        stage = args.stage
        stage_registry = registry[f"stage2_{stage}"]
        bundles = [SeedBundle(**item) for item in stage_registry["seed_bundles"]]
        rows = build_phase2_rows(
            stage=stage,
            arms=_select_arms(args.arms),
            seed_bundles=bundles,
            class_draw_seeds=stage_registry["new_class_draw_seeds"],
            git_commit=args.git_commit,
        )
    logical_rows = len(rows)
    unique_physical_row_count = (
        logical_rows if args.phase == "phase1" else None
    )
    return {
        "schema": "cvs.full_ablation.plan.v1",
        "design_id": DESIGN_ID,
        "phase": args.phase,
        "stage": stage,
        "git_commit": args.git_commit,
        "seed_registry_path": str(registry_path),
        "seed_registry_sha256": registry_hash,
        "wisig_pkl_sha256": str(
            getattr(args, "wisig_pkl_sha256", "")
        ).strip().lower(),
        "registered_phase1_train_seeds": (
            registered_phase1_train_seeds
        ),
        "registered_stage2_method_seeds": (
            [int(bundle.method_seed) for bundle in bundles]
            if args.phase == "phase2"
            else []
        ),
        "stage2_seed_disjointness_verified": (
            args.phase == "phase2"
        ),
        "python_environment_id": str(
            getattr(args, "python_environment_id", "CVS-RFFI")
        ).strip(),
        "formal_launch_authority": False,
        "release_gate": "LOCAL_T0_AND_INDEPENDENT_P0_P1_REVIEW_REQUIRED",
        "logical_row_count": logical_rows,
        "unique_physical_row_count": unique_physical_row_count,
        "physical_dedup_status": (
            "NOT_APPLICABLE_PHASE1"
            if args.phase == "phase1"
            else "PENDING_EFFECTIVE_CONFIG_AND_INPUT_BINDING"
        ),
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a non-launching full-ablation matrix."
    )
    parser.add_argument("--phase", choices=("phase1", "phase2"), required=True)
    parser.add_argument(
        "--phase1-matrix",
        choices=("t1", "label"),
        default="t1",
        help="Phase1 only: main T1 matrix or P1-LABEL sensitivity matrix.",
    )
    parser.add_argument(
        "--stage",
        choices=("screening", "confirmation"),
        default="screening",
    )
    parser.add_argument(
        "--arms",
        default="t1",
        help="Phase2 only: t1 or comma-separated registered arm IDs.",
    )
    parser.add_argument("--git-commit", required=True)
    parser.add_argument(
        "--wisig-pkl-sha256",
        default="",
        help=(
            "Phase1 only: SHA256 of the immutable WiSig pickle; "
            "required before sealing."
        ),
    )
    parser.add_argument(
        "--python-environment-id",
        default="CVS-RFFI",
        help="Verified remote Conda environment basename.",
    )
    parser.add_argument(
        "--seed-registry",
        default=str(DEFAULT_SEED_REGISTRY),
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite plan: {output}")
    plan = build_plan(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
    output.write_text(payload + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "logical_row_count": plan["logical_row_count"],
                "unique_physical_row_count": plan["unique_physical_row_count"],
                "formal_launch_authority": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
