from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build_full_ablation_plan import (
    _canonical_text_sha256,
    build_plan,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT / "configs" / "full_ablation_20260728" / "seed_registry.json"
)
LABEL_REFERENCE = (
    ROOT
    / "configs"
    / "full_ablation_20260728"
    / "phase1_label_rho100_reference_v1.json"
)


def _args(
    *,
    phase: str,
    stage: str = "screening",
    phase1_matrix: str = "t1",
    phase2_matrix: str = "stage2c",
    arms: str = "P2-FULL",
) -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        stage=stage,
        phase1_matrix=phase1_matrix,
        phase1_label_reference=str(LABEL_REFERENCE),
        phase2_matrix=phase2_matrix,
        arms=arms,
        git_commit="a" * 40,
        wisig_pkl_sha256="b" * 64,
        python_environment_id="CVS-RFFI",
        seed_registry=str(REGISTRY),
    )


def test_phase2_plan_does_not_claim_static_physical_dedup() -> None:
    plan = build_plan(_args(phase="phase2"))
    assert plan["logical_row_count"] == 75
    assert plan["unique_physical_row_count"] is None
    assert (
        plan["physical_dedup_status"]
        == "PENDING_EFFECTIVE_CONFIG_AND_INPUT_BINDING"
    )
    assert plan["stage2_seed_disjointness_verified"] is True
    assert plan["registered_stage2_method_seeds"] == [
        7282101,
        7282102,
        7282103,
    ]
    assert all(
        row["method_seed"] == row["train_seed"]
        and row["phase1_bundle_training_seed"] is None
        for row in plan["rows"]
    )


def test_phase2_state_plan_is_independent_from_stage2c_dimensions() -> None:
    plan = build_plan(
        _args(
            phase="phase2",
            phase2_matrix="states",
            arms="t1",
        )
    )
    assert plan["phase2_matrix"] == "states"
    assert plan["stage"] == "state_confirmation"
    assert plan["logical_row_count"] == 325
    assert {
        row["phase"] for row in plan["rows"]
    } == {"stage2a", "stage2b"}
    assert not any(
        row["new_class_count"] or row["new_class_draw_seed"] is not None
        for row in plan["rows"]
    )


def test_phase1_plan_keeps_exact_physical_count() -> None:
    plan = build_plan(_args(phase="phase1"))
    assert plan["logical_row_count"] == 30
    assert plan["unique_physical_row_count"] == 30
    assert plan["physical_dedup_status"] == "NOT_APPLICABLE_PHASE1"
    assert plan["python_environment_id"] == "CVS-RFFI"


def test_phase1_label_plan_has_fourteen_new_rows_and_reuses_rho10() -> None:
    plan = build_plan(
        _args(phase="phase1", phase1_matrix="label")
    )
    assert plan["stage"] == "label"
    assert plan["logical_row_count"] == 14
    assert plan["unique_physical_row_count"] == 14
    assert {row["rho_label"] for row in plan["rows"]} == {
        0.005,
        0.01,
        0.02,
        0.05,
    }
    assert (
        plan["phase1_label_reference"]["source_run_id"]
        == "cvs_full_ablation_phase1_t1_20260729_v5_reuse"
    )
    assert len(plan["phase1_label_reference"]["rows"]) == 5
    assert len(plan["phase1_label_reference_sha256"]) == 64


def test_registry_hash_is_cross_platform_line_ending_stable() -> None:
    lf = b'{"schema":"example"}\n'
    crlf = b'{"schema":"example"}\r\n'
    assert _canonical_text_sha256(lf) == _canonical_text_sha256(crlf)
