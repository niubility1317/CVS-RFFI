from __future__ import annotations

import argparse
from pathlib import Path

from cvsrffi.full_ablation_spec import PHASE2_E0_256_JOINT_ABLATION_ARMS
from cvsrffi.stage2_ablation_factory import (
    STAGE2_E0_256_INTERACTION_ARMS,
    resolve_stage2_config,
    stage2_config_diff,
    validate_stage2_catalog,
)
from scripts.build_full_ablation_plan import build_plan


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "full_ablation_20260728" / "seed_registry.json"
LABEL_REFERENCE = (
    ROOT
    / "configs"
    / "full_ablation_20260728"
    / "phase1_label_rho100_reference_v1.json"
)


EXPECTED_NEW_IDS = (
    "P2-256-J-B0-C3",
    "P2-256-J-B0-D0",
    "P2-256-J-B0-D2",
    "P2-256-J-C3-D0",
    "P2-256-J-C3-D2",
    "P2-256-J-B0-C3-D0",
    "P2-256-J-B0-C3-D2",
)

EXPECTED_MATRIX_IDS = (
    "P2-256-FULL",
    "P2-256-B0",
    "P2-256-C3",
    "P2-256-J-B0-C3",
    "P2-256-D0",
    "P2-256-J-B0-D0",
    "P2-256-J-C3-D0",
    "P2-256-J-B0-C3-D0",
    "P2-256-D2",
    "P2-256-J-B0-D2",
    "P2-256-J-C3-D2",
    "P2-256-J-B0-C3-D2",
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        phase="phase2",
        stage="screening",
        phase1_matrix="t1",
        phase1_label_reference=str(LABEL_REFERENCE),
        phase2_matrix="e0_256_joint_screen",
        arms="t1",
        git_commit="a" * 40,
        wisig_pkl_sha256="b" * 64,
        python_environment_id="CVS-RFFI",
        seed_registry=str(REGISTRY),
        receiver_id="3-19",
        k_shot=10,
        new_class_count=5,
        method_seed=7282101,
        new_class_draw_seed=7282401,
    )


def test_joint_catalog_is_exactly_the_nonredundant_bc_by_geometry_surface() -> None:
    assert tuple(spec.ablation_id for spec in STAGE2_E0_256_INTERACTION_ARMS) == EXPECTED_NEW_IDS
    assert tuple(spec.ablation_id for spec in PHASE2_E0_256_JOINT_ABLATION_ARMS) == EXPECTED_MATRIX_IDS
    assert all("D0-D2" not in spec.ablation_id for spec in PHASE2_E0_256_JOINT_ABLATION_ARMS)
    assert all("F0" not in spec.ablation_id for spec in PHASE2_E0_256_JOINT_ABLATION_ARMS)
    validate_stage2_catalog()


def test_joint_configs_declare_and_resolve_exact_combined_differences() -> None:
    expected = {
        "P2-256-J-B0-C3": ("center_profile", "covariance_profile"),
        "P2-256-J-B0-D0": ("center_profile", "geometry_profile"),
        "P2-256-J-B0-D2": ("center_profile", "geometry_profile"),
        "P2-256-J-C3-D0": ("covariance_profile", "geometry_profile"),
        "P2-256-J-C3-D2": ("covariance_profile", "geometry_profile"),
        "P2-256-J-B0-C3-D0": (
            "center_profile",
            "covariance_profile",
            "geometry_profile",
        ),
        "P2-256-J-B0-C3-D2": (
            "center_profile",
            "covariance_profile",
            "geometry_profile",
        ),
    }
    for ablation_id, keys in expected.items():
        assert tuple(stage2_config_diff(ablation_id)) == keys
        config = resolve_stage2_config(ablation_id)
        assert config["feature_profile"] == "identity160_fft96_beta4_blocknorm_globalnorm"
        assert config["quantization_profile"] == (
            "f3_dual_residual_int8_fp16_block_scale_bias_fp16_diag_metric"
        )


def test_joint_plan_has_one_matched_twelve_arm_screen() -> None:
    plan = build_plan(_args())
    assert plan["phase2_matrix"] == "e0_256_joint_screen"
    assert plan["logical_row_count"] == 12
    assert tuple(row["ablation_id"] for row in plan["rows"]) == EXPECTED_MATRIX_IDS
    assert {row["receiver_id"] for row in plan["rows"]} == {"3-19"}
    assert {(row["k_shot"], row["new_class_count"]) for row in plan["rows"]} == {
        (10, 5)
    }
    assert {row["new_class_draw_seed"] for row in plan["rows"]} == {7282401}
