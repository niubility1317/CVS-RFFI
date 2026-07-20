import numpy as np
import pytest

from cvsrffi.stage2_d96_ra_cgsrda import (
    D96RACGSRDAError,
    Phase1LockedConfig,
    SRDAState,
    build_ground_domain_model,
    fit_srda,
    fuse_base_srda_logits,
)


def _config() -> Phase1LockedConfig:
    return Phase1LockedConfig(
        tau=0.2,
        ridge=0.05,
        temp_base=1.0,
        temp_aux=1.0,
        max_rank=3,
        phase1_receipt_sha256="1" * 64,
    )


def _ground_grid(domains: int = 6, classes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(9601)
    class_axes = rng.normal(size=(classes, 160))
    class_axes /= np.linalg.norm(class_axes, axis=1, keepdims=True)
    domain_axes = rng.normal(scale=0.08, size=(domains, 160))
    return class_axes[None, :, :] + domain_axes[:, None, :]


def test_ground_builder_rejects_partially_populated_domain() -> None:
    grid = _ground_grid()
    mask = np.ones(grid.shape[:2], dtype=bool)
    mask[-1, -1] = False
    with pytest.raises(D96RACGSRDAError, match="complete domain grids"):
        build_ground_domain_model(grid, mask, _config())


def test_ground_builder_is_redundancy_aware_and_rank_bounded() -> None:
    grid = _ground_grid()
    model = build_ground_domain_model(
        grid, np.ones(grid.shape[:2], dtype=bool), _config()
    )
    assert model.nuisance_basis_fp32.shape[0] == 160
    assert model.nuisance_basis_fp32.shape[1] <= 3
    assert np.sum(model.density_weights_fp32) == pytest.approx(1.0)
    assert model.audit["ground_class_score_access"] is False
    assert model.audit["query_rows_used"] == 0
    assert np.all(model.nuisance_eigenvalues_fp32 > 0)


def test_fit_uses_target_means_for_every_registered_class() -> None:
    rng = np.random.default_rng(9602)
    grid = _ground_grid()
    ground = build_ground_domain_model(
        grid, np.ones(grid.shape[:2], dtype=bool), _config()
    )
    labels = np.repeat(np.asarray([10, 11, 12, 20]), 3)
    support = rng.normal(size=(len(labels), 160))
    support += np.repeat(grid[0, :3], 3, axis=0).tolist() + [np.zeros(160)] * 3
    state = fit_srda(ground, support, labels, np.asarray([10, 11, 12]), _config())
    expected = []
    normalized = support / np.linalg.norm(support, axis=1, keepdims=True)
    for label in state.classes:
        expected.append(normalized[labels == label].mean(axis=0))
    np.testing.assert_allclose(state.target_means_fp32, expected, atol=1e-6)
    assert state.audit["target_class_mean_source"].startswith("registered_target_support")
    assert state.audit["ground_class_mean_used_for_score"] is False
    assert state.logits(normalized[:2]).shape == (2, 4)


def test_k1_and_zero_weight_return_exact_base_object() -> None:
    base = np.asarray([[0.1, 0.9]], dtype=np.float32)
    aux = np.asarray([[2.0, -1.0]], dtype=np.float32)
    def state(k_shot: int, rho: float) -> SRDAState:
        return SRDAState(
            classes=(10, 11),
            coefficient_fp32=np.zeros((2, 160), dtype=np.float32),
            intercept_fp32=np.zeros(2, dtype=np.float32),
            target_means_fp32=np.zeros((2, 160), dtype=np.float32),
            coverage_rho=rho,
            k_shot=k_shot,
            config_lock_digest=_config().lock_digest,
            audit={},
        )
    fused, audit = fuse_base_srda_logits(
        base,
        aux,
        base_classes=(10, 11),
        srda_state=state(1, 0.8),
        support_cv_reliability=1.0,
        reliability_source="support_crossfit_phase1_smoothed",
        support_cv_receipt_sha256="6" * 64,
        config=_config(),
    )
    assert fused is base
    assert audit["k1_forced_base_fallback"] is True
    fused, audit = fuse_base_srda_logits(
        base,
        aux,
        base_classes=(10, 11),
        srda_state=state(10, 0.0),
        support_cv_reliability=1.0,
        reliability_source="support_crossfit_phase1_smoothed",
        support_cv_receipt_sha256="6" * 64,
        config=_config(),
    )
    assert fused is base
    assert audit["w0_exact_base_fallback"] is True


def test_phase1_lock_and_psd_inputs_are_hard_failures() -> None:
    with pytest.raises(D96RACGSRDAError):
        Phase1LockedConfig(0.2, 0.05, 1.0, 1.0, 3, "not-a-sha")
    with pytest.raises(D96RACGSRDAError):
        Phase1LockedConfig(0.2, 0.05, 1.0, 1.0, 5, "1" * 64)
    with pytest.raises(D96RACGSRDAError):
        Phase1LockedConfig(0.2, 0.05, 1.0, 1.0, 1.9, "1" * 64)


def test_nonzero_d96_reliability_requires_sealed_support_cv_receipt() -> None:
    state = SRDAState(
        classes=(10, 11),
        coefficient_fp32=np.zeros((2, 160), dtype=np.float32),
        intercept_fp32=np.zeros(2, dtype=np.float32),
        target_means_fp32=np.zeros((2, 160), dtype=np.float32),
        coverage_rho=0.5,
        k_shot=10,
        config_lock_digest=_config().lock_digest,
        audit={},
    )
    with pytest.raises(D96RACGSRDAError, match="reliability drift"):
        fuse_base_srda_logits(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            base_classes=(10, 11),
            srda_state=state,
            support_cv_reliability=0.5,
            reliability_source="support_crossfit_phase1_smoothed",
            support_cv_receipt_sha256=None,
            config=_config(),
        )


def test_ground_fit_rejects_config_drift_and_reports_real_fp32_state() -> None:
    rng = np.random.default_rng(9603)
    grid = _ground_grid()
    ground = build_ground_domain_model(grid, np.ones(grid.shape[:2], bool), _config())
    changed = Phase1LockedConfig(0.2, 0.06, 1.0, 1.0, 3, "1" * 64)
    labels = np.repeat(np.asarray([10, 11, 12]), 2)
    with pytest.raises(D96RACGSRDAError, match="differs"):
        fit_srda(ground, rng.normal(size=(6, 160)), labels, np.asarray([10, 11, 12]), changed)
    state = fit_srda(
        ground, rng.normal(size=(6, 160)), labels, np.asarray([10, 11, 12]), _config()
    )
    actual = state.coefficient_fp32.nbytes + state.intercept_fp32.nbytes + state.target_means_fp32.nbytes
    assert state.audit["compiled_aux_state_dtype"] == "fp32"
    assert state.audit["compiled_aux_fp32_state_bytes"] == actual
