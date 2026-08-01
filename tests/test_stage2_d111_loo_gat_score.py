from __future__ import annotations

from dataclasses import replace
import inspect
from types import MappingProxyType

import numpy as np
import pytest

from cvsrffi.stage2_d111_loo_gat_bundle import D111Bundle, FEATURE_DIM
from cvsrffi.stage2_d111_loo_gat_score import (
    D111ScoreError,
    WEISZFELD_STEPS,
    audit_d111_loo_gat_state,
    fit_d111_loo_gat_state,
    predict_d111_loo_gat,
    score_d111_loo_gat_logits,
    _weiszfeld_certificate,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


OLD = tuple(f"old-{index}" for index in range(6))
CLASSES = OLD + ("new-0",)


def _readonly(value: np.ndarray, dtype=np.float32) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


def _lock(k: int, *, gamma: float = 1.0) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=3.0,
        kernel_effective_dim=FEATURE_DIM,
        kernel_volume_gamma=gamma,
        shared_h0=0.2,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _geometry() -> tuple[np.ndarray, np.ndarray]:
    anchors = np.zeros((6, FEATURE_DIM), dtype=np.float32)
    for index in range(6):
        anchors[index, 10 + index] = 1.0
    basis = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    basis[np.arange(3), np.arange(3)] = 1.0
    return anchors, basis


def _bundle(*, envelope_b: float = 0.03, epsilon: float = 0.01) -> D111Bundle:
    anchors, basis = _geometry()
    return D111Bundle(
        class_registry=OLD,
        anchors=_readonly(anchors),
        basis=_readonly(basis),
        v_g=_readonly(np.full(6, 0.002, dtype=np.float32)),
        v_s=0.002,
        envelope_b=envelope_b,
        epsilon=epsilon,
        manifest=MappingProxyType(
            {
                "effective_formal_phase2_eligible": True,
                "effective_bundle_state": "FORMAL_D111_OUTER_JOINT_SEALED",
                "content_root_sha256": "a" * 64,
            }
        ),
    )


def _support(k: int, *, shared: bool = True) -> tuple[np.ndarray, list[str]]:
    anchors, basis = _geometry()
    common = np.asarray([0.04, -0.02, 0.01], dtype=np.float32)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, name in enumerate(OLD):
        if shared:
            shift = common
        else:
            shift = np.asarray(
                [0.025 * class_index, -0.02 * (class_index % 3), 0.015 * (-1) ** class_index],
                dtype=np.float32,
            )
        vector = anchors[class_index] + basis.T @ shift
        vector /= np.linalg.norm(vector)
        for _ in range(k):
            rows.append(vector.copy())
            labels.append(name)
    new = np.zeros(FEATURE_DIM, dtype=np.float32)
    new[30] = 1.0
    for _ in range(k):
        rows.append(new.copy())
        labels.append("new-0")
    return np.asarray(rows, dtype=np.float32), labels


def _bank(k: int, *, shared: bool = True):
    support, labels = _support(k, shared=shared)
    return build_typed_zid_support_bank(support, labels, CLASSES, config=_lock(k))


def test_qualified_k1_is_nonidentity_and_new_class_mass_is_zero() -> None:
    bank = _bank(1)
    state = fit_d111_loo_gat_state(_bundle(), bank)
    old = np.asarray(state.old_class_indices)
    assert np.all(state.qualified[old])
    assert np.all(state.iterations[old] == WEISZFELD_STEPS)
    assert np.all(state.gap[old] <= _bundle().epsilon)
    assert np.all(state.consensus_count[old] >= 3)
    assert np.all((state.rho[old] > 0.0) & (state.rho[old] < 1.0))
    assert state.rho[CLASSES.index("new-0")] == 0.0
    decoded_support, _labels = _support(1)
    assert np.linalg.norm(state.anchors[0] - decoded_support[0]) > 1.0e-7


def test_ineligible_state_is_exact_m0_fallback() -> None:
    bank = _bank(1, shared=False)
    state = fit_d111_loo_gat_state(_bundle(envelope_b=1.0e-8, epsilon=1.0e-12), bank)
    assert not np.any(state.qualified)
    assert not np.any(state.rho)
    query, _ = _support(1, shared=False)
    expected = score_zid_student_t_logits(
        bank, query, metric=identity_shared_psd_metric(config=bank.config)
    )
    actual = score_d111_loo_gat_logits(state, bank, query)
    assert np.array_equal(actual, expected)


def test_unit_mass_formula_reuses_m0_density_and_bandwidth() -> None:
    bank = _bank(5)
    state = fit_d111_loo_gat_state(_bundle(), bank)
    query, _ = _support(5)
    query = query[:3]
    baseline = score_zid_student_t_logits(
        bank, query, metric=identity_shared_psd_metric(config=bank.config)
    )
    actual = score_d111_loo_gat_logits(state, bank, query)
    class_index = state.old_class_indices[0]
    rho = float(state.rho[class_index])
    anchor = state.anchors[class_index].astype(np.float64)
    normalized_query = query.astype(np.float64) / np.linalg.norm(query, axis=1, keepdims=True)
    distance = np.maximum(2.0 * (1.0 - np.clip(normalized_query @ anchor, -1.0, 1.0)), 0.0)
    h = float(bank.class_scales_fp16[class_index])
    anchor_log = (
        -bank.config.kernel_volume_gamma * FEATURE_DIM * np.log(h)
        - 0.5
        * (bank.config.student_nu + FEATURE_DIM)
        * np.log1p(distance / (bank.config.student_nu * h * h))
    )
    expected = np.logaddexp(
        np.log1p(-rho) + baseline[:, class_index].astype(np.float64),
        np.log(rho) + anchor_log,
    ).astype(np.float32)
    np.testing.assert_array_equal(actual[:, class_index], expected)
    np.testing.assert_array_equal(actual[:, -1], baseline[:, -1])


def test_unit_mass_rejects_non_normalized_m0_volume_exponent() -> None:
    support, labels = _support(1)
    bank = build_typed_zid_support_bank(
        support, labels, CLASSES, config=_lock(1, gamma=0.75)
    )
    with pytest.raises(D111ScoreError, match="kernel_volume_gamma=1"):
        fit_d111_loo_gat_state(_bundle(), bank)


def test_query_order_and_batch_splitting_do_not_change_state_or_scores() -> None:
    bank = _bank(1)
    state = fit_d111_loo_gat_state(_bundle(), bank)
    rng = np.random.default_rng(9)
    query = rng.normal(size=(11, FEATURE_DIM)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    receipt = state.state_receipt_sha256
    anchors_before = state.anchors.copy()
    full = score_d111_loo_gat_logits(state, bank, query)
    split = np.concatenate(
        [
            score_d111_loo_gat_logits(state, bank, query[:4]),
            score_d111_loo_gat_logits(state, bank, query[4:]),
        ],
        axis=0,
    )
    order = np.asarray([7, 1, 10, 0, 5, 2, 8, 3, 9, 4, 6])
    permuted = score_d111_loo_gat_logits(state, bank, query[order])
    np.testing.assert_array_equal(full, split)
    np.testing.assert_array_equal(full[order], permuted)
    assert state.state_receipt_sha256 == receipt
    np.testing.assert_array_equal(state.anchors, anchors_before)


def test_coordinate_permutation_equivariance_and_class_permutation_symmetry() -> None:
    bank = _bank(1)
    bundle = _bundle()
    first = fit_d111_loo_gat_state(bundle, bank)

    coordinate_order = np.roll(np.arange(FEATURE_DIM), 7)
    support, labels = _support(1)
    transformed_bank = build_typed_zid_support_bank(
        support[:, coordinate_order], labels, CLASSES, config=_lock(1)
    )
    transformed_bundle = D111Bundle(
        class_registry=bundle.class_registry,
        anchors=_readonly(bundle.anchors[:, coordinate_order]),
        basis=_readonly(bundle.basis[:, coordinate_order]),
        v_g=bundle.v_g,
        v_s=bundle.v_s,
        envelope_b=bundle.envelope_b,
        epsilon=bundle.epsilon,
        manifest=bundle.manifest,
    )
    transformed = fit_d111_loo_gat_state(transformed_bundle, transformed_bank)
    np.testing.assert_allclose(first.rho, transformed.rho, atol=1.0e-7, rtol=0.0)
    np.testing.assert_allclose(
        first.anchors[:, coordinate_order], transformed.anchors, atol=1.0e-6, rtol=0.0
    )

    class_order = (3, 0, 5, 1, 4, 2)
    permuted_old = tuple(OLD[index] for index in class_order)
    permuted_classes = permuted_old + ("new-0",)
    permuted_rows = []
    permuted_labels = []
    for name in permuted_classes:
        mask = np.asarray(labels) == name
        permuted_rows.extend(support[mask])
        permuted_labels.extend([name] * int(np.sum(mask)))
    permuted_bank = build_typed_zid_support_bank(
        np.asarray(permuted_rows, dtype=np.float32),
        permuted_labels,
        permuted_classes,
        config=_lock(1),
    )
    permuted_bundle = D111Bundle(
        class_registry=permuted_old,
        anchors=_readonly(bundle.anchors[list(class_order)]),
        basis=bundle.basis,
        v_g=_readonly(bundle.v_g[list(class_order)]),
        v_s=bundle.v_s,
        envelope_b=bundle.envelope_b,
        epsilon=bundle.epsilon,
        manifest=bundle.manifest,
    )
    permuted = fit_d111_loo_gat_state(permuted_bundle, permuted_bank)
    inverse = [permuted_classes.index(name) for name in CLASSES]
    np.testing.assert_allclose(first.rho, permuted.rho[inverse], atol=0.0, rtol=0.0)


def test_dense_orthogonal_equivariance_at_decoded_real_geometry_layer() -> None:
    rng = np.random.default_rng(44)
    residuals = rng.normal(size=(5, 3))
    rotation3, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    centre, primal, dual, gap, feasible = _weiszfeld_certificate(residuals)
    rotated_centre, rotated_primal, rotated_dual, rotated_gap, rotated_feasible = (
        _weiszfeld_certificate(residuals @ rotation3)
    )
    np.testing.assert_allclose(centre @ rotation3, rotated_centre, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        [primal, dual, gap],
        [rotated_primal, rotated_dual, rotated_gap],
        atol=1.0e-11,
        rtol=0.0,
    )
    assert feasible is rotated_feasible

    q = rng.normal(size=(4, FEATURE_DIM))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    anchor = rng.normal(size=FEATURE_DIM)
    anchor /= np.linalg.norm(anchor)
    rotation160, _ = np.linalg.qr(rng.normal(size=(FEATURE_DIM, FEATURE_DIM)))
    np.testing.assert_allclose(
        q @ anchor,
        (q @ rotation160) @ (anchor @ rotation160),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_state_is_readonly_bound_and_resource_is_query_independent() -> None:
    bank = _bank(1)
    state = fit_d111_loo_gat_state(_bundle(), bank)
    for value in (
        state.anchors,
        state.rho,
        state.qualified,
        state.iterations,
        state.primal,
        state.dual,
        state.gap,
        state.consensus_count,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    audit = audit_d111_loo_gat_state(state)
    resource = audit["resource_receipt"]
    assert resource["persistent_numeric_bytes"] == 4711
    assert resource["enrollment_projection_macs"] == 2880
    assert resource["weiszfeld_scalar_steps"] == 2880
    assert resource["extra_query_macs_per_row_upper_bound"] == 960
    assert resource["query_dependent_state_bytes"] == 0
    assert audit["query_rows_used_for_fit"] == 0
    with pytest.raises(TypeError):
        audit["old_class_count"] = 9
    broken = replace(state, state_receipt_sha256="0" * 64)
    query, _ = _support(1)
    with pytest.raises(D111ScoreError, match="receipt"):
        score_d111_loo_gat_logits(broken, bank, query[:1])


def test_api_has_no_truth_role_quota_or_query_fit_surface() -> None:
    fit_parameters = set(inspect.signature(fit_d111_loo_gat_state).parameters)
    assert fit_parameters == {"bundle", "bank"}
    for forbidden in ("truth", "role", "quota", "receiver", "query"):
        assert not any(forbidden in name.lower() for name in fit_parameters)
    score_parameters = set(inspect.signature(score_d111_loo_gat_logits).parameters)
    assert score_parameters == {"state", "bank", "query_zid"}
    bank = _bank(1)
    state = fit_d111_loo_gat_state(_bundle(), bank)
    query, _ = _support(1)
    predicted = predict_d111_loo_gat(state, bank, query[:2])
    assert len(predicted) == 2
    assert all(label in CLASSES for label in predicted)
