from __future__ import annotations

from cvsrffi.stage2_ablation_factory import (
    STAGE2_E0_256_ABLATION_ARMS,
    resolve_stage2_config,
    stage2_config_diff,
    validate_stage2_catalog,
)


EXPECTED_IDS = (
    "P2-256-FULL",
    "P2-256-A0",
    "P2-256-B0",
    "P2-256-S0",
    "P2-256-C3",
    "P2-256-D0",
    "P2-256-D2",
)


def test_current_256d_module_ablation_catalog_is_closed_and_excludes_f0() -> None:
    """The approved current-method screen has no FP32 F0 comparator."""

    assert tuple(spec.ablation_id for spec in STAGE2_E0_256_ABLATION_ARMS) == EXPECTED_IDS
    assert all("F0" not in spec.ablation_id for spec in STAGE2_E0_256_ABLATION_ARMS)
    assert all(
        spec.reference_id == "P2-256-FULL"
        for spec in STAGE2_E0_256_ABLATION_ARMS
    )
    validate_stage2_catalog()


def test_current_256d_controls_change_one_declared_method_field() -> None:
    full = resolve_stage2_config("P2-256-FULL")
    assert full["feature_profile"] == "identity160_fft96_beta4_blocknorm_globalnorm"
    assert stage2_config_diff("P2-256-FULL") == {}
    expected = {
        "P2-256-A0": "feature_profile",
        "P2-256-B0": "center_profile",
        "P2-256-S0": "covariance_profile",
        "P2-256-C3": "covariance_profile",
        "P2-256-D0": "geometry_profile",
        "P2-256-D2": "geometry_profile",
    }
    for ablation_id, key in expected.items():
        assert tuple(stage2_config_diff(ablation_id)) == (key,)
