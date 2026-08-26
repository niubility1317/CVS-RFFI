from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_ablation_factory import (
    STAGE2_E0_256_ABLATION_ARMS,
    STAGE2_E0_256_INTERACTION_ARMS,
    STAGE2_T1_ARMS,
)
from cvsrffi.stage2_ablation_executors import (
    _fit_with_fp32_centering_audit,
    _project_feature_profile,
    fit_stage2_ablation,
)


def _fixture(k_shot: int = 2):
    rng = np.random.default_rng(940)
    old_classes = tuple(f"old-{index}" for index in range(6))
    new_classes = tuple(f"new-{index}" for index in range(5))
    means = rng.normal(size=(11, 288)).astype(np.float32)
    old_targets = np.repeat(np.arange(6), k_shot)
    new_targets = np.repeat(np.arange(5), k_shot)
    old_rows = (
        means[old_targets]
        + 0.05 * rng.normal(size=(6 * k_shot, 288))
    ).astype(np.float32)
    new_rows = (
        means[6 + new_targets]
        + 0.05 * rng.normal(size=(5 * k_shot, 288))
    ).astype(np.float32)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    ground = {
        "ground_basis": basis,
        "ground_spectral_weights": np.asarray([0.5, 0.3, 0.2]),
        "ground_audit": {
            "d81_basis_sha256": "a" * 64,
            "d81_spectral_weight_sha256": "b" * 64,
            "d81_participation_ratio_effective_rank": 2.6,
            "d81_retained_rank": 3,
            "d81_rank_policy": "ceil_participation_ratio_effective_rank",
            "ground_component_input_count": 84,
            "ground_statistic_semantics": (
                "class_centered_cross_domain_centroid_drift_eigenspectrum"
            ),
        },
    }
    return {
        "old_support_features": old_rows,
        "old_support_labels": np.asarray(old_classes)[old_targets],
        "old_classes": old_classes,
        "new_support_features": new_rows,
        "new_support_labels": np.asarray(new_classes)[new_targets],
        "new_classes": new_classes,
        **ground,
    }


def _full_registered_feature_rows(row_count: int = 4) -> np.ndarray:
    rng = np.random.default_rng(9411)
    identity = rng.normal(size=(row_count, 160)).astype(np.float32)
    fft = rng.normal(size=(row_count, 96)).astype(np.float32)
    rf = rng.normal(size=(row_count, 32)).astype(np.float32)

    def normalize(rows: np.ndarray) -> np.ndarray:
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    primary = normalize(identity)
    auxiliary = normalize(np.concatenate([normalize(fft), normalize(rf)], axis=1))
    return normalize(np.concatenate([primary, 4.0 * auxiliary], axis=1))


def test_feature_profile_projection_reconstructs_compact_fft_and_legacy_rf_geometry() -> None:
    full = _full_registered_feature_rows()
    primary = full[:, :160] * np.sqrt(17.0)
    auxiliary = full[:, 160:] * (np.sqrt(17.0) / 4.0)

    expected_fft_auxiliary = auxiliary[:, :96] / np.linalg.norm(
        auxiliary[:, :96], axis=1, keepdims=True
    )
    expected_fft = np.concatenate(
        [
            primary / np.linalg.norm(primary, axis=1, keepdims=True),
            4.0 * expected_fft_auxiliary,
        ],
        axis=1,
    )
    expected_fft /= np.linalg.norm(expected_fft, axis=1, keepdims=True)

    expected_rf_auxiliary = np.concatenate(
        [
            np.zeros((len(full), 96), dtype=np.float32),
            auxiliary[:, 96:]
            / np.linalg.norm(auxiliary[:, 96:], axis=1, keepdims=True),
        ],
        axis=1,
    )
    expected_rf = np.concatenate(
        [
            primary / np.linalg.norm(primary, axis=1, keepdims=True),
            4.0 * expected_rf_auxiliary,
        ],
        axis=1,
    )
    expected_rf /= np.linalg.norm(expected_rf, axis=1, keepdims=True)

    np.testing.assert_allclose(
        _project_feature_profile(full, "identity160_fft96_beta4_blocknorm_globalnorm"),
        expected_fft,
        rtol=2e-6,
        atol=2e-6,
    )
    assert expected_fft.shape == (len(full), 256)
    np.testing.assert_allclose(
        _project_feature_profile(full, "identity160_rf32_beta4_blocknorm_globalnorm"),
        expected_rf,
        rtol=2e-6,
        atol=2e-6,
    )


def test_fit_api_has_no_query_surface() -> None:
    signature = inspect.signature(fit_stage2_ablation)
    assert not any("query" in name.lower() for name in signature.parameters)


def test_fp32_centering_roundoff_policy_is_explicitly_audited() -> None:
    def fit(rows, targets, class_count, k_shot):
        return (
            np.zeros((class_count, rows.shape[1]), dtype=np.float32),
            np.zeros(class_count, dtype=np.float32),
            {"k_shot": k_shot, "target_count": len(targets)},
        )

    coefficient, intercept, audit = _fit_with_fp32_centering_audit(
        fit,
        np.ones((2, 3), dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        2,
        1,
    )
    assert coefficient.shape == (2, 3)
    assert intercept.shape == (2,)
    assert audit["stage2_ablation_fp32_centering_argmax_drift_allowed"] is True

    def failing_fit(*_args):
        raise RuntimeError("synthetic fit failure")

    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        _fit_with_fp32_centering_audit(
            failing_fit,
            np.ones((2, 3), dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            2,
            1,
        )


@pytest.mark.parametrize(
    "ablation_id",
    (
        "P2-BASE-COSINE",
        "P2-BASE-EUCLIDEAN",
        "P2-BASE-QKNN",
        "P2-BASE-DIAG-LDA",
        "P2-BASE-POOLED-LW-LDA",
    ),
)
def test_closed_form_baselines_fit_and_predict_all_registered_classes(
    ablation_id: str,
) -> None:
    fixture = _fixture(2)
    state = fit_stage2_ablation(
        ablation_id=ablation_id,
        seed=820001,
        device="cpu",
        **fixture,
    )
    query = np.concatenate(
        [fixture["old_support_features"][:2], fixture["new_support_features"][:2]]
    )
    scores = state.score(query)
    assert scores.shape == (4, 11)
    assert state.predict(query).shape == (4,)
    assert state.audit["query_rows_used"] == 0


def test_stage2b_proto_has_old_registry_only() -> None:
    fixture = _fixture(2)
    state = fit_stage2_ablation(
        ablation_id="P2-S2B-PROTO",
        old_support_features=fixture["old_support_features"],
        old_support_labels=fixture["old_support_labels"],
        old_classes=fixture["old_classes"],
        seed=820001,
    )
    assert state.stage == "stage2b"
    assert len(state.classes) == 6
    assert state.score(fixture["old_support_features"][:3]).shape == (3, 6)


@pytest.mark.parametrize(
    "ablation_id",
    tuple(spec.ablation_id for spec in STAGE2_T1_ARMS),
)
def test_every_frozen_t1_arm_has_a_reachable_numerical_executor(
    ablation_id: str,
) -> None:
    fixture = _fixture(2)
    spec = next(
        item for item in STAGE2_T1_ARMS
        if item.ablation_id == ablation_id
    )
    if spec.stage == "stage2a":
        labels = np.asarray(fixture["old_support_labels"])
        prototypes = np.stack(
            [
                fixture["old_support_features"][labels == class_handle].mean(
                    axis=0
                )
                for class_handle in fixture["old_classes"]
            ]
        )
        state = fit_stage2_ablation(
            ablation_id=ablation_id,
            old_support_features=None,
            old_support_labels=None,
            old_classes=fixture["old_classes"],
            deployment_prototypes=prototypes,
            seed=820001,
        )
        query = fixture["old_support_features"][:2]
        expected_classes = 6
    elif spec.stage == "stage2b":
        state = fit_stage2_ablation(
            ablation_id=ablation_id,
            old_support_features=fixture["old_support_features"],
            old_support_labels=fixture["old_support_labels"],
            old_classes=fixture["old_classes"],
            ground_basis=fixture["ground_basis"],
            ground_spectral_weights=fixture["ground_spectral_weights"],
            ground_audit=fixture["ground_audit"],
            seed=820001,
            device="cpu",
        )
        query = fixture["old_support_features"][:2]
        expected_classes = 6
    else:
        state = fit_stage2_ablation(
            ablation_id=ablation_id,
            seed=820001,
            device="cpu",
            **fixture,
        )
        query = fixture["new_support_features"][:2]
        expected_classes = 11
    assert state.score(query).shape == (2, expected_classes)
    assert state.predict(query).shape == (2,)
    assert state.audit["query_rows_used"] == 0


def test_full_k2_uses_exact_low_k_fallback_and_all_class_argmax() -> None:
    fixture = _fixture(2)
    state = fit_stage2_ablation(
        ablation_id="P2-FULL",
        seed=820001,
        device="cpu",
        **fixture,
    )
    assert state.audit["d92_status"] == "k1_k2_exact_d81_fallback"
    assert state.audit["d92_registration_balanced_active"] is False
    assert state.score_kind == "compiled_affine"
    assert state.compiled_affine_state is not None
    assert state.compiled_affine_state.arm_id == "P2-F3"
    assert state.coefficient_fp32.size == 0
    assert state.intercept_fp32.size == 0
    assert state.resource["deployment_claim"] == "storage_compression_only"
    query = np.concatenate(
        [fixture["old_support_features"][:1], fixture["new_support_features"][:1]]
    )
    assert state.score(query).shape == (2, 11)
    assert state.predict(query).shape == (2,)


def test_current_256d_s0_reaches_active_fixed_ridge_registration_fit() -> None:
    fixture = _fixture(5)
    state = fit_stage2_ablation(
        ablation_id="P2-256-S0",
        seed=820001,
        device="cpu",
        **fixture,
    )
    assert state.ablation_id == "P2-256-S0"
    assert state.feature_profile == "identity160_fft96_beta4_blocknorm_globalnorm"
    assert state.audit["d92_registration_balanced_active"] is True
    assert state.audit["d92_covariance_mode"] == "empirical_fixed_ridge"
    assert state.compiled_affine_state is not None
    assert state.compiled_affine_state.arm_id == "P2-F3"
    assert state.compiled_affine_state.feature_dim == 256
    assert state.compiled_affine_state.block_offsets == (0, 160, 256)


@pytest.mark.parametrize(
    ("ablation_id", "expected_feature_dim", "expected_block_offsets"),
    (
        ("P2-256-FULL", 256, (0, 160, 256)),
        ("P2-256-A0", 160, (0, 160)),
    ),
)
def test_current_256d_arms_compile_only_their_active_feature_coordinates(
    ablation_id: str,
    expected_feature_dim: int,
    expected_block_offsets: tuple[int, ...],
) -> None:
    """Catch zero-padding that silently turns the approved 256D path into 288D."""

    fixture = _fixture(2)
    state = fit_stage2_ablation(
        ablation_id=ablation_id,
        seed=820001,
        device="cpu",
        **fixture,
    )
    compiled = state.compiled_affine_state
    assert compiled is not None
    assert compiled.feature_dim == expected_feature_dim
    assert compiled.block_offsets == expected_block_offsets


def test_current_256d_catalogue_is_numerically_reachable_without_f0() -> None:
    fixture = _fixture(2)
    for spec in STAGE2_E0_256_ABLATION_ARMS:
        state = fit_stage2_ablation(
            ablation_id=spec.ablation_id,
            seed=820001,
            device="cpu",
            **fixture,
        )
        assert state.ablation_id == spec.ablation_id
        assert state.score(fixture["new_support_features"][:2]).shape == (2, 11)


@pytest.mark.parametrize(
    ("ablation_id", "expected_geometry", "expects_bias_normalization"),
    (
        ("P2-256-J-B0-C3-D0", "full_only", True),
        ("P2-256-J-B0-C3-D2", "full_block_fixed_half", False),
    ),
)
def test_current_256d_joint_endpoints_compose_support_only_profiles(
    ablation_id: str,
    expected_geometry: str,
    expects_bias_normalization: bool,
) -> None:
    """The compatible three-factor endpoints must execute, not merely exist in a plan."""

    fixture = _fixture(5)
    state = fit_stage2_ablation(
        ablation_id=ablation_id,
        seed=820001,
        device="cpu",
        **fixture,
    )

    compiled = state.compiled_affine_state
    assert compiled is not None
    assert compiled.arm_id == "P2-F3"
    assert compiled.feature_dim == 256
    assert compiled.block_offsets == (0, 160, 256)
    assert state.score(fixture["new_support_features"][:2]).shape == (2, 11)
    assert state.audit["query_rows_used"] == 0
    assert state.resource["query_rows_used_for_fit"] == 0
    assert state.audit["d81_probe_arm"] == "ground_nuisance_cauchy_center"
    assert state.audit["d81_ground_int8_component_used"] is False
    assert bool(state.audit["affine_common_bias_normalized_for_fp16"]) is (
        expects_bias_normalization
    )
    assert state.audit[
        "affine_common_bias_normalization_support_argmax_preserved"
    ] is True
    assert state.audit["joint_profiles"] == {
        "center_profile": "support_plain_mean_no_ground_spectrum",
        "covariance_profile": "d81_all_classes_equal_ledoit_wolf",
        "geometry_profile": expected_geometry,
    }


@pytest.mark.parametrize(
    "ablation_id",
    tuple(spec.ablation_id for spec in STAGE2_E0_256_INTERACTION_ARMS),
)
def test_every_current_256d_joint_arm_reaches_one_f3_support_only_state(
    ablation_id: str,
) -> None:
    fixture = _fixture(2)
    state = fit_stage2_ablation(
        ablation_id=ablation_id,
        seed=820001,
        device="cpu",
        **fixture,
    )

    compiled = state.compiled_affine_state
    assert compiled is not None
    assert compiled.arm_id == "P2-F3"
    assert compiled.feature_dim == 256
    assert state.audit["query_rows_used"] == 0
    assert state.resource["query_rows_used_for_fit"] == 0
    assert state.score(fixture["new_support_features"][:2]).shape == (2, 11)


def test_td_htrc_m21_is_reachable_as_an_explicit_opt_in_mode() -> None:
    fixture = _fixture(2)
    rng = np.random.default_rng(941)
    state = fit_stage2_ablation(
        ablation_id="P2-E0",
        seed=820001,
        device="cpu",
        ground_class_centers=rng.normal(size=(6, 160)),
        module2_mode="td_htrc_m21",
        **fixture,
    )
    assert state.audit["module2_mode"] == "td_htrc_m21"
    assert state.audit["td_htrc_method"] == "TD-HTRC-M2.1"
    assert state.audit["td_htrc_query_rows_used"] == 0
    assert state.resource["td_htrc_query_extra_macs"] == 0
    assert state.score(fixture["new_support_features"][:2]).shape == (2, 11)


def test_td_htrc_m22_is_reachable_and_keeps_query_path_unchanged() -> None:
    fixture = _fixture(5)
    rng = np.random.default_rng(942)
    ground_full = rng.normal(size=(6, 288))
    state = fit_stage2_ablation(
        ablation_id="P2-E0",
        seed=820001,
        device="cpu",
        ground_class_centers=ground_full[:, :160],
        ground_full_centers=ground_full,
        module2_mode="td_htrc_m22",
        **fixture,
    )
    assert state.audit["module2_mode"] == "td_htrc_m22"
    assert state.audit["td_htrc_method"] == "TD-HTRC-M2.2"
    assert state.audit["td_htrc_m22_query_rows_used"] == 0
    assert state.resource["td_htrc_query_extra_macs"] == 0
    assert state.resource["td_htrc_registration_macs"] > 0
    assert state.score(fixture["new_support_features"][:2]).shape == (2, 11)


def test_adapter_head_baseline_trains_a_real_support_only_adapter() -> None:
    fixture = _fixture(2)
    state = fit_stage2_ablation(
        ablation_id="P2-BASE-ADAPTER-HEAD",
        seed=820001,
        device="cpu",
        **fixture,
    )
    assert state.score_kind == "adapter_cosine_affine"
    assert state.adapter_u_fp32.shape == (288, 8)
    assert state.adapter_v_fp32.shape == (288, 8)
    assert state.adapter_gate_fp32.shape == (8,)
    assert state.audit["adapter_support_only"] is True
    assert state.audit["adapter_query_rows_used"] == 0
    assert state.audit["adapter_epochs"] == 12
    assert len(state.training_trace) == 12
    assert state.resource["optimizer_steps"] == 12
    assert state.resource["trainable_parameters"] == 4616
    assert state.score(
        fixture["new_support_features"][:2]
    ).shape == (2, 11)


@pytest.mark.parametrize(
    ("ablation_id", "expected_dtype", "expected_layers"),
    (
        ("P2-F0", np.float32, 1),
        ("P2-F1", np.float16, 1),
        ("P2-F2", np.int8, 1),
        ("P2-F3", np.int8, 2),
    ),
)
def test_quantization_arms_persist_only_the_selected_compiled_head(
    ablation_id: str,
    expected_dtype: np.dtype,
    expected_layers: int,
) -> None:
    fixture = _fixture(2)
    state = fit_stage2_ablation(
        ablation_id=ablation_id,
        seed=820001,
        device="cpu",
        **fixture,
    )
    compiled = state.compiled_affine_state
    assert compiled is not None
    assert compiled.arm_id == ablation_id
    assert len(compiled.coefficient_layers) == expected_layers
    assert compiled.coefficient_layers[0].dtype == expected_dtype
    assert compiled.has_fp32_coefficient_sidecar is False
    assert state.coefficient_fp32.size == 0
    assert state.intercept_fp32.size == 0
    assert state.resource["persistent_head_state_bytes"] == compiled.state_bytes
    assert state.score(fixture["old_support_features"][:1]).shape == (1, 11)
