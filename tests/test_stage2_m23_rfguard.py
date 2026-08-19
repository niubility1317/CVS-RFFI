from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np

from cvsrffi.stage2_m23_rfguard import (
    ARM_RF_LITE_DIAG,
    ARM_RF_LITE_GATED,
    ARM_RF_QUALITY,
    FFT_DIM,
    IDENTITY_DIM,
    IF_DIM,
    M23Config,
    M23CenterEstimate,
    RF_LITE_DIM,
    build_ground_manifold,
    build_rfguard_blocks,
    estimate_stage2b_domain_state,
    extract_rf_lite_quality,
    fit_rfguard_m23,
)


OLD_CLASSES = tuple(f"old_{index}" for index in range(6))
NEW_CLASSES = tuple(f"new_{index}" for index in range(5))


def _normalise(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class _FakeGroundComponent:
    def __init__(self, *, seed: int = 23) -> None:
        rng = np.random.default_rng(seed)
        self.domain_registry = tuple(f"domain_{index}" for index in range(5))
        self.class_registry = OLD_CLASSES
        core = _normalise(rng.normal(size=(len(OLD_CLASSES), IDENTITY_DIM)))
        shared_u = _normalise(rng.normal(size=(2, IDENTITY_DIM)))
        dense: list[np.ndarray] = []
        for domain_index in range(len(self.domain_registry)):
            shared = (
                0.06 * np.sin(domain_index + 1.0) * shared_u[0]
                + 0.04 * np.cos(domain_index + 0.5) * shared_u[1]
            )
            rows = []
            for class_index in range(len(OLD_CLASSES)):
                interaction = np.zeros(IDENTITY_DIM, dtype=np.float64)
                interaction[32 + class_index] = (
                    0.025 * (domain_index - 2) * (-1.0 if class_index % 2 else 1.0)
                )
                rows.append(core[class_index] + shared + interaction)
            dense.append(_normalise(np.stack(rows)))
        self._dense = np.stack(dense).astype(np.float32)
        self.manifest = {
            "resource_audit": {
                "reconstruction_rmse": 1.0e-3,
                "logical_deployment_state_bytes": 4096,
            }
        }

    def reconstruct_domain(self, domain_handle: str) -> np.ndarray:
        return np.array(
            self._dense[self.domain_registry.index(str(domain_handle))], copy=True
        )

    def resource_audit(self) -> dict[str, float | int]:
        return dict(self.manifest["resource_audit"])


def _legacy288_and_rf_lite(*, rows: int = 8, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    zid = _normalise(rng.normal(size=(rows, IDENTITY_DIM)))
    fft = _normalise(rng.normal(size=(rows, FFT_DIM)))
    legacy_rf = _normalise(rng.normal(size=(rows, 32)))
    legacy = np.concatenate(
        [zid / np.sqrt(17.0), 4.0 * fft / np.sqrt(34.0), 4.0 * legacy_rf / np.sqrt(34.0)],
        axis=1,
    ).astype(np.float32)
    lite = _normalise(rng.normal(size=(rows, RF_LITE_DIM))).astype(np.float32)
    return legacy, lite


def _support(
    component: _FakeGroundComponent,
    *,
    k_shot: int,
    informative_rf: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ground = np.mean(component._dense, axis=0)
    new_identity = _normalise(rng.normal(size=(len(NEW_CLASSES), IDENTITY_DIM)))
    old_fft = _normalise(rng.normal(size=(len(OLD_CLASSES), FFT_DIM)))
    new_fft = _normalise(rng.normal(size=(len(NEW_CLASSES), FFT_DIM)))
    old_rf = _normalise(rng.normal(size=(len(OLD_CLASSES), RF_LITE_DIM)))
    new_rf = _normalise(rng.normal(size=(len(NEW_CLASSES), RF_LITE_DIM)))

    def rows_for(
        identity_centres: np.ndarray,
        fft_centres: np.ndarray,
        rf_centres: np.ndarray,
    ) -> np.ndarray:
        rows = []
        for class_index in range(len(identity_centres)):
            for _ in range(k_shot):
                zid = _normalise(
                    identity_centres[class_index]
                    + 0.018 * rng.normal(size=IDENTITY_DIM)
                )
                fft = _normalise(
                    fft_centres[class_index] + 0.025 * rng.normal(size=FFT_DIM)
                )
                rf = (
                    _normalise(
                        rf_centres[class_index]
                        + 0.035 * rng.normal(size=RF_LITE_DIM)
                    )
                    if informative_rf
                    else _normalise(rng.normal(size=RF_LITE_DIM))
                )
                rows.append(np.concatenate([zid, fft, rf]))
        return np.asarray(rows, dtype=np.float32)

    old_rows = rows_for(ground, old_fft, old_rf)
    new_rows = rows_for(new_identity, new_fft, new_rf)
    old_labels = np.repeat(np.asarray(OLD_CLASSES), k_shot)
    new_labels = np.repeat(np.asarray(NEW_CLASSES), k_shot)
    old_quality = np.ones(len(old_rows), dtype=np.float32)
    new_quality = np.ones(len(new_rows), dtype=np.float32)
    if k_shot > 1:
        old_quality[k_shot - 1] = 0.1
        old_rows[k_shot - 1, :IDENTITY_DIM] = _normalise(
            rng.normal(size=IDENTITY_DIM)
        )
    return old_rows, old_labels, old_quality, new_rows, new_labels, new_quality


def _received_iq(*, anomalous: bool) -> np.ndarray:
    rng = np.random.default_rng(99)
    symbol = rng.choice(np.asarray([1.0, -1.0]), size=2048) + 1j * rng.choice(
        np.asarray([1.0, -1.0]), size=2048
    )
    value = symbol + 0.04 * (
        rng.normal(size=symbol.size) + 1j * rng.normal(size=symbol.size)
    )
    if anomalous:
        value = value + (1.8 + 1.1j)
        value[::9] *= 5.0
        value = np.clip(value.real, -2.0, 2.0) + 1j * np.clip(
            value.imag, -2.0, 2.0
        )
    return np.stack([value.real, value.imag], axis=0)[None, :].astype(np.float32)


def test_rf_lite_is_gain_invariant_finite_and_quality_downweights_anomalies() -> None:
    clean = _received_iq(anomalous=False)
    bad = _received_iq(anomalous=True)
    clean_lite, clean_quality = extract_rf_lite_quality(clean)
    scaled_lite, scaled_quality = extract_rf_lite_quality(7.5 * clean)
    bad_lite, bad_quality = extract_rf_lite_quality(bad)

    assert clean_lite.shape == bad_lite.shape == (1, RF_LITE_DIM)
    assert np.isfinite(clean_lite).all() and np.isfinite(bad_lite).all()
    assert np.allclose(clean_lite, scaled_lite, atol=2e-5)
    assert np.allclose(clean_quality, scaled_quality, atol=2e-5)
    assert 0.1 <= float(bad_quality[0]) < float(clean_quality[0]) <= 1.0


def test_rf_lite_correlation_coordinates_ignore_cfo_phase_rotation() -> None:
    clean = _received_iq(anomalous=False)
    samples = clean.shape[-1]
    phase = np.exp(1j * np.linspace(0.0, 3.0 * np.pi, samples))
    complex_clean = clean[0, 0] + 1j * clean[0, 1]
    rotated = complex_clean * phase
    rotated_iq = np.stack([rotated.real, rotated.imag], axis=0)[None, :].astype(
        np.float32
    )
    clean_lite, _ = extract_rf_lite_quality(clean)
    rotated_lite, _ = extract_rf_lite_quality(rotated_iq)
    assert np.allclose(clean_lite[:, -2:], rotated_lite[:, -2:], atol=5e-3)


def test_compact_blocks_recover_if_geometry_without_288_zero_padding() -> None:
    legacy, lite = _legacy288_and_rf_lite()
    blocks = build_rfguard_blocks(legacy, lite)
    assert blocks.shape == (len(legacy), IDENTITY_DIM + FFT_DIM + RF_LITE_DIM)
    assert np.allclose(np.linalg.norm(blocks[:, :IDENTITY_DIM], axis=1), 1.0)
    assert np.allclose(
        np.linalg.norm(blocks[:, IDENTITY_DIM:IF_DIM], axis=1), 1.0
    )
    assert np.allclose(np.linalg.norm(blocks[:, IF_DIM:], axis=1), 1.0)
    assert not np.any(np.all(blocks[:, IF_DIM:] == 0.0, axis=0))


def test_ground_manifold_uses_true_domain_class_bank_and_class_loo() -> None:
    component = _FakeGroundComponent()
    manifold = build_ground_manifold(component)
    assert manifold.class_centres.shape == (len(OLD_CLASSES), IDENTITY_DIM)
    assert manifold.shared_basis.shape[0] == IDENTITY_DIM
    assert manifold.interaction_basis_by_class.shape[0] == len(OLD_CLASSES)
    for class_index, source_indices in enumerate(
        manifold.loo_source_class_indices
    ):
        assert class_index not in source_indices
        assert set(source_indices) == set(range(len(OLD_CLASSES))) - {class_index}
    assert manifold.audit["domain_class_cell_count"] == 5 * len(OLD_CLASSES)
    assert manifold.audit["class_loo_interaction_enabled"] is True


def test_stage2b_domain_state_keeps_out_of_manifold_offset() -> None:
    component = _FakeGroundComponent()
    old, labels, quality, *_ = _support(
        component, k_shot=5, informative_rf=True, seed=31
    )
    manifold = build_ground_manifold(component)
    shifted = np.array(old, copy=True)
    direction = np.zeros(IDENTITY_DIM, dtype=np.float32)
    direction[-1] = 0.08
    shifted[:, :IDENTITY_DIM] = _normalise(
        shifted[:, :IDENTITY_DIM] + direction
    )
    state = estimate_stage2b_domain_state(
        shifted,
        labels,
        OLD_CLASSES,
        quality,
        manifold,
    )
    assert state.out_of_manifold_offset_norm > 0.0
    assert np.linalg.norm(state.shared_offset) > 0.0
    assert state.nuisance_covariance.shape == (IDENTITY_DIM, IDENTITY_DIM)
    assert float(np.linalg.eigvalsh(state.nuisance_covariance).min()) >= -1e-10


def test_stage2b_domain_state_uses_class_loo_interaction_basis() -> None:
    component = _FakeGroundComponent()
    old, labels, quality, *_ = _support(
        component, k_shot=5, informative_rf=True, seed=37
    )
    manifold = build_ground_manifold(component)
    shifted = np.array(old, copy=True)
    for class_index, item in enumerate(OLD_CLASSES):
        mask = labels == item
        direction = manifold.interaction_basis_by_class[class_index, :, 0]
        shifted[mask, :IDENTITY_DIM] = _normalise(
            shifted[mask, :IDENTITY_DIM]
            + (0.12 if class_index % 2 == 0 else -0.12) * direction
        )
    state = estimate_stage2b_domain_state(
        shifted,
        labels,
        OLD_CLASSES,
        quality,
        manifold,
    )
    without_interaction = estimate_stage2b_domain_state(
        shifted,
        labels,
        OLD_CLASSES,
        quality,
        replace(
            manifold,
            interaction_basis_by_class=np.zeros_like(
                manifold.interaction_basis_by_class
            ),
        ),
    )
    assert state.class_interaction_offsets.shape == (
        len(OLD_CLASSES),
        IDENTITY_DIM,
    )
    assert np.linalg.norm(state.class_interaction_offsets) > 0.0
    assert not np.allclose(state.target_centres, without_interaction.target_centres)
    assert state.audit["class_loo_interaction_applied"] is True


def test_k1_forces_256d_if_head_without_rf_or_loo() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=1, informative_rf=True, seed=44
    )
    domain = estimate_stage2b_domain_state(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        build_ground_manifold(component),
    )
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_LITE_GATED,
        frozen_domain_state=domain,
    )
    assert state.compiled_affine_state.feature_dim == IF_DIM == 256
    assert state.audit["m23_k_regime"] == "K1_IF_PROTOTYPE_DIAG"
    assert state.audit["m23_support_loo_enabled"] is False
    assert max(state.audit["m23_rf_gate_by_class"]) == 0.0
    assert state.audit["m23_rf_cross_block_frobenius"] == 0.0
    assert state.audit["m23_covariance_min_eigenvalue"] > 0.0


def test_k2_forces_diagonal_if_head_and_zero_rf_gate() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=2, informative_rf=True, seed=47
    )
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_LITE_GATED,
    )
    assert state.compiled_affine_state.feature_dim == IF_DIM
    assert state.audit["m23_k_regime"] == "K2_IF_TASK_DIAG"
    assert state.audit["m23_covariance_structure"] == "diagonal_if"
    assert max(state.audit["m23_rf_gate_by_class"]) == 0.0


def test_k10_reuses_frozen_stage2b_domain_state_and_compiles_266d_f3() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=10, informative_rf=True, seed=52
    )
    domain = estimate_stage2b_domain_state(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        build_ground_manifold(component),
    )
    old_copy = np.array(old, copy=True)
    new_copy = np.array(new, copy=True)
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_LITE_GATED,
        frozen_domain_state=domain,
    )
    assert state.compiled_affine_state.feature_dim == IF_DIM + RF_LITE_DIM == 266
    assert state.compiled_affine_state.block_offsets == (0, 160, 256, 266)
    assert state.domain_state.digest == domain.digest
    assert state.audit["m23_stage2b_domain_state_reused"] is True
    assert state.audit["m23_center_override_enabled"] is True
    assert state.audit["m23_new_class_ground_prior_count"] == 0
    assert state.audit["m23_old_class_ground_prior_count"] == len(OLD_CLASSES)
    assert state.audit["m23_rf_cross_block_frobenius"] == 0.0
    assert state.audit["m23_covariance_min_eigenvalue"] > 0.0
    assert 0.0 <= min(state.audit["m23_rf_gate_by_class"])
    assert max(state.audit["m23_rf_gate_by_class"]) <= 1.0
    assert np.array_equal(old, old_copy) and np.array_equal(new, new_copy)

    assert not hasattr(state, "fp32_coefficient")
    assert not hasattr(state, "fp32_bias")
    assert not hasattr(state, "score_fp32_reference")
    assert state.audit["m23_has_fp32_coefficient_sidecar"] is False
    assert state.audit["m23_quantization_support_prediction_agreement_rate"] >= 0.99
    fp32_bytes = len(state.classes) * (IF_DIM + RF_LITE_DIM + 1) * 4
    assert state.compiled_affine_state.state_bytes < fp32_bytes
    assert state.audit["m23_total_retained_state_bytes"] >= (
        state.compiled_affine_state.state_bytes
    )


def test_center_uncertainty_is_class_specific_and_penalises_intercept() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=5, informative_rf=True, seed=61
    )
    rng = np.random.default_rng(62)
    first_new = np.flatnonzero(new_y == NEW_CLASSES[0])
    new[first_new, :IDENTITY_DIM] = _normalise(
        rng.normal(size=(len(first_new), IDENTITY_DIM))
    )
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_QUALITY,
    )
    penalties = np.asarray(
        state.audit["m23_center_uncertainty_intercept_penalty_by_class"]
    )
    assert penalties.shape == (len(OLD_CLASSES) + len(NEW_CLASSES),)
    assert np.all(penalties >= 0.0)
    assert penalties[len(OLD_CLASSES)] > np.median(penalties[: len(OLD_CLASSES)])


def test_module2_returns_explicit_centres_uncertainty_weights_and_domain_covariance() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=5, informative_rf=True, seed=64
    )
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_LITE_DIAG,
    )
    estimate = state.center_estimate
    assert isinstance(estimate, M23CenterEstimate)
    assert estimate.centres.shape == (
        len(OLD_CLASSES) + len(NEW_CLASSES),
        IDENTITY_DIM + FFT_DIM + RF_LITE_DIM,
    )
    assert estimate.centre_uncertainty.shape == estimate.centres.shape
    assert estimate.support_weights.shape == (len(old) + len(new),)
    assert np.isclose(np.sum(estimate.support_weights[: len(old)]), 0.5)
    assert np.isclose(np.sum(estimate.support_weights[len(old) :]), 0.5)
    assert estimate.domain_nuisance_covariance.shape == (IDENTITY_DIM, IDENTITY_DIM)
    assert estimate.ground_prior_mask[: len(OLD_CLASSES), :IDENTITY_DIM].all()
    assert not estimate.ground_prior_mask[len(OLD_CLASSES) :].any()
    assert state.audit["m23_identity_weight"] == 1.0
    assert state.audit["m23_fft_weight"] == 4.0
    assert state.audit["m23_covariance_task_weight"] == {"old": 0.5, "new": 0.5}
    assert state.audit["m23_support_reliability_rule"] == (
        "rf_quality_times_inverse_one_plus_if_residual_over_class_tau"
    )
    assert 0.0 <= state.audit["m23_rf_coefficient_frobenius_ratio"] <= 1.0


def test_rf_complexity_penalty_and_no_harm_can_fall_back_to_base() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=10, informative_rf=False, seed=70
    )
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_LITE_GATED,
        config=replace(M23Config(), rf_complexity_penalty=100.0),
    )
    assert max(state.audit["m23_rf_gate_by_class"]) < 1e-6
    assert state.audit["m23_rf_no_harm_fallback_count"] >= 0


def test_k5_rf_gate_is_one_global_decision_for_every_class() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=5, informative_rf=True, seed=74
    )
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_LITE_GATED,
    )
    gates = np.asarray(state.audit["m23_rf_gate_by_class"])
    assert np.array_equal(gates, np.full_like(gates, gates[0]))
    assert state.audit["m23_rf_gate_mode"] == "global_support_loo"
    assert state.audit["m23_rf_global_help"] == sum(
        state.audit["m23_rf_loo_help_by_class"]
    )
    assert state.audit["m23_rf_global_harm"] == sum(
        state.audit["m23_rf_loo_harm_by_class"]
    )


def test_rf_lite_diag_has_psd_addition_and_zero_cross_block() -> None:
    component = _FakeGroundComponent()
    old, old_y, old_q, new, new_y, new_q = _support(
        component, k_shot=5, informative_rf=True, seed=77
    )
    state = fit_rfguard_m23(
        old,
        old_y,
        OLD_CLASSES,
        old_q,
        ground_component=component,
        new_support_blocks=new,
        new_support_labels=new_y,
        new_classes=NEW_CLASSES,
        new_support_quality=new_q,
        arm=ARM_RF_LITE_DIAG,
    )
    assert state.audit["m23_covariance_structure"] == "if_full_plus_rf_diag"
    assert state.audit["m23_rf_cross_block_frobenius"] == 0.0
    assert state.audit["m23_nuisance_covariance_min_eigenvalue"] >= 0.0
    assert state.audit["m23_covariance_min_eigenvalue"] > 0.0


def test_fit_surface_has_no_query_or_truth_parameter() -> None:
    parameters = inspect.signature(fit_rfguard_m23).parameters
    forbidden = ("query", "truth", "quota", "receiver", "scene")
    assert not any(
        token in name.lower() for name in parameters for token in forbidden
    )
