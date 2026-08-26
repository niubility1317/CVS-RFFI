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
    PHASE2_E0_256_ABLATION_ARMS,
    PHASE2_E0_256_JOINT_ABLATION_ARMS,
    PHASE2_STATE_T1_ARMS,
    PHASE2_T1_ARMS,
    ArmSpec,
    FullAblationSpecError,
    SeedBundle,
    build_phase1_label_rows,
    build_phase1_t1_rows,
    build_phase2_e0_256_joint_screen_rows,
    build_phase2_e0_256_screen_rows,
    build_phase2_rows,
    build_phase2_state_rows,
    validate_stage2_registry_disjointness,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_REGISTRY = (
    ROOT / "configs" / "full_ablation_20260728" / "seed_registry.json"
)
DEFAULT_PHASE1_LABEL_REFERENCE = (
    ROOT
    / "configs"
    / "full_ablation_20260728"
    / "phase1_label_rho100_reference_v1.json"
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


def _load_phase1_label_reference(
    path: Path,
    registered_seeds: list[int],
) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    reference = json.loads(payload.decode("utf-8-sig"))
    if (
        reference.get("schema")
        != "cvs.full_ablation.phase1_label_reference.v1"
        or reference.get("design_id") != DESIGN_ID
        or reference.get("ablation_id") != "P1-FULL"
        or abs(float(reference.get("rho_label", -1.0)) - 0.10) > 1e-12
        or reference.get("reuse_mode")
        != "reference_only_not_dispatched"
        or reference.get("required_before_label_curve_analysis") is not True
    ):
        raise FullAblationSpecError(
            "invalid Phase1 rho=0.10 reference manifest"
        )
    if not str(reference.get("source_run_id", "")).strip():
        raise FullAblationSpecError(
            "Phase1 label reference lacks source_run_id"
        )
    if not str(reference.get("source_run_root", "")).strip() or not str(
        reference.get("source_log_root", "")
    ).strip():
        raise FullAblationSpecError(
            "Phase1 label reference lacks source run/log roots"
        )
    rows = list(reference.get("rows") or [])
    expected_rows = {
        f"P1-FULL__train_seed_{seed}": int(seed)
        for seed in registered_seeds
    }
    actual_rows = {
        str(row.get("row_key", "")): int(row.get("train_seed", -1))
        for row in rows
    }
    if len(rows) != 5 or actual_rows != expected_rows:
        raise FullAblationSpecError(
            "Phase1 rho=0.10 reference must bind five registered P1-FULL rows"
        )
    required_artifacts = {
        "best_source_validation_ssdg.pth",
        "phase1_training_completion_receipt.json",
        "phase1_terminal_status.json",
        "phase1_resource_summary.json",
        "frozen_phase1_heldout_eval.json",
        "phase2_zid_prototypes.pt",
        "phase2_zid_prototypes.json",
    }
    if set(reference.get("expected_artifacts") or []) != required_artifacts:
        raise FullAblationSpecError(
            "Phase1 rho=0.10 reference artifact contract drift"
        )
    return reference, _canonical_text_sha256(payload)


def _select_arms(
    raw_ids: str, arm_space: tuple[ArmSpec, ...] = PHASE2_T1_ARMS
) -> list[ArmSpec]:
    available = {arm.ablation_id: arm for arm in arm_space}
    if raw_ids.strip().lower() == "t1":
        return list(arm_space)
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
            label_reference, label_reference_hash = (
                _load_phase1_label_reference(
                    Path(args.phase1_label_reference).resolve(),
                    registered_phase1_train_seeds,
                )
            )
            rows = build_phase1_label_rows(
                registered_phase1_train_seeds,
                git_commit=args.git_commit,
            )
            stage = "label"
        else:
            label_reference = {}
            label_reference_hash = ""
            rows = build_phase1_t1_rows(
                registered_phase1_train_seeds,
                git_commit=args.git_commit,
            )
            stage = "t1"
    else:
        label_reference = {}
        label_reference_hash = ""
        registered_phase1_train_seeds = []
        phase2_matrix = str(
            getattr(args, "phase2_matrix", "stage2c")
        ).strip()
        if phase2_matrix not in {
            "stage2c",
            "states",
            "e0_256_screen",
            "e0_256_joint_screen",
        }:
            raise FullAblationSpecError("unknown Phase2 matrix")
        if phase2_matrix in {
            "e0_256_screen",
            "e0_256_joint_screen",
        } and args.stage != "screening":
            raise FullAblationSpecError(
                "current-256D screens are screening-only matrices"
            )
        registry_stage = (
            "confirmation" if phase2_matrix == "states" else args.stage
        )
        stage_registry = registry[f"stage2_{registry_stage}"]
        bundles = [SeedBundle(**item) for item in stage_registry["seed_bundles"]]
        if phase2_matrix == "states":
            stage = "state_confirmation"
            rows = build_phase2_state_rows(
                arms=_select_arms(args.arms, PHASE2_STATE_T1_ARMS),
                seed_bundles=bundles,
                git_commit=args.git_commit,
            )
        elif phase2_matrix in {"e0_256_screen", "e0_256_joint_screen"}:
            joint_screen = phase2_matrix == "e0_256_joint_screen"
            selected = _select_arms(
                args.arms,
                (
                    PHASE2_E0_256_JOINT_ABLATION_ARMS
                    if joint_screen
                    else PHASE2_E0_256_ABLATION_ARMS
                ),
            )
            method_seed = int(getattr(args, "method_seed", 7282101))
            selected_bundles = [
                bundle
                for bundle in bundles
                if int(bundle.method_seed) == method_seed
            ]
            if len(selected_bundles) != 1:
                raise FullAblationSpecError(
                    "current-256D screen requires one registered method seed"
                )
            draw_seed = int(
                getattr(args, "new_class_draw_seed", 7282401)
            )
            registered_draws = [
                int(value)
                for value in stage_registry["new_class_draw_seeds"]
            ]
            if draw_seed not in registered_draws:
                raise FullAblationSpecError(
                    "current-256D screen draw seed is not in the registry"
                )
            bundles = selected_bundles
            row_builder = (
                build_phase2_e0_256_joint_screen_rows
                if joint_screen
                else build_phase2_e0_256_screen_rows
            )
            rows = row_builder(
                arms=selected,
                seed_bundle=bundles[0],
                class_draw_seed=draw_seed,
                receiver_id=str(getattr(args, "receiver_id", "3-19")),
                k_shot=int(getattr(args, "k_shot", 10)),
                new_class_count=int(
                    getattr(args, "new_class_count", 5)
                ),
                git_commit=args.git_commit,
            )
            stage = "screening"
        else:
            stage = args.stage
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
        "phase2_matrix": (
            str(getattr(args, "phase2_matrix", "stage2c"))
            if args.phase == "phase2"
            else None
        ),
        "git_commit": args.git_commit,
        "seed_registry_path": str(registry_path),
        "seed_registry_sha256": registry_hash,
        "wisig_pkl_sha256": str(
            getattr(args, "wisig_pkl_sha256", "")
        ).strip().lower(),
        "registered_phase1_train_seeds": (
            registered_phase1_train_seeds
        ),
        "phase1_label_reference_path": (
            str(Path(args.phase1_label_reference).resolve())
            if args.phase == "phase1" and stage == "label"
            else ""
        ),
        "phase1_label_reference_sha256": label_reference_hash,
        "phase1_label_reference": label_reference,
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
        "--phase1-label-reference",
        default=str(DEFAULT_PHASE1_LABEL_REFERENCE),
        help="Phase1 label only: machine-readable rho=0.10 P1-FULL reference.",
    )
    parser.add_argument(
        "--stage",
        choices=("screening", "confirmation"),
        default="screening",
    )
    parser.add_argument(
        "--phase2-matrix",
        choices=(
            "stage2c",
            "states",
            "e0_256_screen",
            "e0_256_joint_screen",
        ),
        default="stage2c",
        help=(
            "Phase2 only: Stage2-C registration rows, independent Stage2-A/B "
            "state tables, the approved current-256D same-row screen, or its "
            "B0×C3×geometry interaction screen."
        ),
    )
    parser.add_argument(
        "--arms",
        default="t1",
        help=(
            "Phase2 only: t1 or comma-separated registered arm IDs. "
            "For e0_256_screen and e0_256_joint_screen, t1 means the exact "
            "approved arm surface."
        ),
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
        "--receiver-id",
        default="3-19",
        help="Current-256D screen only; fixed to the preregistered receiver.",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        default=10,
        help="Current-256D screen only; fixed to the preregistered K-shot.",
    )
    parser.add_argument(
        "--new-class-count",
        type=int,
        default=5,
        help="Current-256D screen only; fixed to the preregistered new count.",
    )
    parser.add_argument(
        "--method-seed",
        type=int,
        default=7282101,
        help="Current-256D screen only; one registered screening bundle.",
    )
    parser.add_argument(
        "--new-class-draw-seed",
        type=int,
        default=7282401,
        help="Current-256D screen only; one registered class draw.",
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
