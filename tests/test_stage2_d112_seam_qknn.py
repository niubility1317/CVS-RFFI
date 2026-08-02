from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from cvsrffi.stage2_d112_seam_bundle import (
    D112BundleError,
    FEATURE_DIM,
    G1_COMPONENT_STATE,
    G1_EVALUATION_SCOPE,
    build_d112_g0_bundle,
    build_d112_source_held_g1_bundle,
)
from cvsrffi.stage2_d112_seam_qknn import (
    fit_d112_ground_head_source_held_g1_state,
    fit_d112_seam_g0_state,
    fit_d112_seam_state,
    fit_d112_seam_source_held_g1_state,
    score_d112_seam_logits,
    score_d112_seam_source_held_g1_logits,
    seam_jacobian_trace,
    sphere_exp,
    sphere_log,
    sphere_parallel_transport,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


OLD = tuple(f"old-{index}" for index in range(6))
CLASSES = OLD + ("new-0",)


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ground = np.zeros((6, FEATURE_DIM), dtype=np.float64)
    for index in range(6):
        ground[index, 10 + index] = 1.0
    q0 = np.sum(ground, axis=0)
    q0 /= np.linalg.norm(q0)
    basis = np.zeros((3, FEATURE_DIM), dtype=np.float64)
    basis[np.arange(3), np.arange(3)] = 1.0
    return ground, q0, basis


def _bundle(*, global_valid: bool = True, order: tuple[int, ...] | None = None):
    ground, q0, basis = _geometry()
    selected = tuple(range(6)) if order is None else order
    return build_d112_g0_bundle(
        class_registry=tuple(OLD[index] for index in selected),
        g=ground[np.asarray(selected)],
        q0=q0,
        U=basis,
        sigma0_r=np.asarray([0.002 + index * 1.0e-5 for index in selected]),
        sigma0_amb=np.asarray([0.002 + index * 1.0e-5 for index in selected]),
        v_g_r=np.asarray([0.001 + index * 1.0e-5 for index in selected]),
        v_g_amb=np.asarray([0.001 + index * 1.0e-5 for index in selected]),
        tau_h_r=0.004,
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        global_bundle_valid=global_valid,
    )


def _source_held_g1_bundle(*, global_valid: bool = True):
    ground, q0, basis = _geometry()
    return build_d112_source_held_g1_bundle(
        class_registry=OLD,
        g=ground,
        q0=q0,
        U=basis,
        sigma0_r=np.asarray([0.002 + index * 1.0e-5 for index in range(6)]),
        sigma0_amb=np.asarray([0.002 + index * 1.0e-5 for index in range(6)]),
        v_g_r=np.asarray([0.001 + index * 1.0e-5 for index in range(6)]),
        v_g_amb=np.asarray([0.001 + index * 1.0e-5 for index in range(6)]),
        tau_h_r=0.004,
        checkpoint_sha256="1" * 64,
        source_aggregate_sha256="2" * 64,
        phase1_seal_sha256="5" * 64,
        source_held_split_sha256="6" * 64,
        global_bundle_valid=global_valid,
    )


def _lock(k: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        k,
        3.0,
        12,
        1.0,
        0.2,
        2.0,
        0.5,
        2.0,
        1.0,
        "3" * 64,
        "4" * 64,
    )


def _support(k: int, *, antipodal: bool = False) -> tuple[np.ndarray, list[str]]:
    ground, _q0, basis = _geometry()
    common = np.asarray([0.04, -0.02, 0.01], dtype=np.float64)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, name in enumerate(OLD):
        vector = -ground[index] if antipodal else ground[index] + basis.T @ common
        vector = vector / np.linalg.norm(vector)
        for _ in range(k):
            rows.append(vector.copy())
            labels.append(name)
    new = np.zeros(FEATURE_DIM, dtype=np.float64)
    new[30] = 1.0
    for _ in range(k):
        rows.append(new.copy())
        labels.append("new-0")
    return np.asarray(rows, dtype=np.float32), labels


def _bank(k: int, *, classes=CLASSES, antipodal: bool = False):
    support, labels = _support(k, antipodal=antipodal)
    return build_typed_zid_support_bank(support, labels, classes, config=_lock(k))


def _m0(bank, query: np.ndarray) -> np.ndarray:
    return score_zid_student_t_logits(
        bank, query, metric=identity_shared_psd_metric(config=bank.config)
    )


def test_bundle_is_readonly_and_content_root_rejects_tamper() -> None:
    bundle = _bundle()
    assert not bundle.g.flags.writeable
    assert not bundle.U.flags.writeable
    tampered = dict(bundle.manifest)
    tampered["content_root_sha256"] = "f" * 64
    with pytest.raises(D112BundleError, match="content root"):
        replace(bundle, manifest=tampered)
    tampered = dict(bundle.manifest)
    tampered["checkpoint_sha256"] = "e" * 64
    with pytest.raises(D112BundleError, match="content root"):
        replace(bundle, manifest=tampered)
    external = dict(bundle.manifest)
    cloned = replace(bundle, manifest=external)
    external["checkpoint_sha256"] = "d" * 64
    assert cloned.manifest["checkpoint_sha256"] == "1" * 64
    assert cloned.manifest is not external


def test_sphere_maps_and_closed_jacobian_are_numerically_consistent() -> None:
    rng = np.random.default_rng(112)
    base = rng.normal(size=FEATURE_DIM)
    base /= np.linalg.norm(base)
    point = rng.normal(size=FEATURE_DIM)
    point /= np.linalg.norm(point)
    tangent = sphere_log(base, point)
    recovered = sphere_exp(base, tangent)
    assert np.allclose(recovered, point, atol=1.0e-12, rtol=0.0)
    transported = sphere_parallel_transport(base, point, tangent)
    assert abs(float(point @ transported)) < 1.0e-12
    assert np.isclose(np.linalg.norm(transported), np.linalg.norm(tangent), atol=1.0e-12)
    assert seam_jacobian_trace(alpha=0.37, uncompressed_norm=0.8) > 0.0


@pytest.mark.parametrize("k", [1, 5, 10])
def test_old_columns_change_but_new_column_is_exact_m0(k: int) -> None:
    bank = _bank(k)
    state = fit_d112_seam_g0_state(_bundle(), bank)
    query, _labels = _support(k)
    baseline = _m0(bank, query)
    actual = score_d112_seam_logits(state, bank, query)
    assert np.all(state.information_valid[np.asarray(state.old_class_indices)])
    assert np.all(state.rho[np.asarray(state.old_class_indices)] > 0.0)
    assert np.all(state.alpha[np.asarray(state.old_class_indices)] > 0.0)
    assert np.all(state.anchor_shift_l2[np.asarray(state.old_class_indices)] > 0.0)
    expected_vh = (
        state.loo_variance_r + state.loo_disagreement_r
    ) * state.jacobian_trace / FEATURE_DIM
    assert np.allclose(state.v_h_amb, expected_vh, atol=1.0e-14, rtol=0.0)
    assert np.max(np.abs(actual[:, :6] - baseline[:, :6])) > 0.0
    assert np.array_equal(actual[:, 6], baseline[:, 6])


@pytest.mark.parametrize("fallback", ["global", "donor"])
def test_global_or_row_geometry_invalid_is_exact_m0(fallback: str) -> None:
    bank = _bank(1, antipodal=fallback == "donor")
    bundle = _bundle(global_valid=fallback != "global")
    state = fit_d112_seam_g0_state(bundle, bank)
    query, _labels = _support(1)
    assert not np.any(state.information_valid)
    assert np.array_equal(score_d112_seam_logits(state, bank, query), _m0(bank, query))


def test_fit_api_has_no_query_truth_role_or_quota_surface() -> None:
    for function in (fit_d112_seam_state, fit_d112_seam_g0_state):
        parameters = set(inspect.signature(function).parameters)
        assert parameters == {"bundle", "bank"}
        assert not parameters & {"query", "truth", "role", "quota", "labels"}


def test_formal_entry_rejects_even_a_global_invalid_g0_bundle() -> None:
    with pytest.raises(Exception, match="target formal D112 Phase2 surface"):
        fit_d112_seam_state(_bundle(global_valid=False), _bank(1))


@pytest.mark.parametrize("k", [1, 5, 10])
def test_source_held_g1_is_immutable_and_leaves_new_class_at_exact_m0(k: int) -> None:
    bundle = _source_held_g1_bundle()
    bank = _bank(k)
    state = fit_d112_seam_source_held_g1_state(bundle, bank)
    query, _labels = _support(k)
    actual = score_d112_seam_source_held_g1_logits(state, bank, query)
    baseline = _m0(bank, query)
    assert bundle.manifest["component_state"] == G1_COMPONENT_STATE
    assert bundle.manifest["evaluation_scope"] == G1_EVALUATION_SCOPE
    assert bundle.manifest["formal_phase2_eligible"] is False
    assert bundle.manifest["target_access_allowed"] is False
    assert bundle.manifest["source_row_runtime_access_allowed"] is False
    assert bundle.manifest["query_truth_access_allowed"] is False
    assert bundle.manifest["source_held_query_access_allowed"] is True
    assert not bundle.g.flags.writeable
    assert state.bundle_component_state == G1_COMPONENT_STATE
    assert state.evaluation_scope == G1_EVALUATION_SCOPE
    assert np.max(np.abs(actual[:, :6] - baseline[:, :6])) > 0.0
    assert np.array_equal(actual[:, 6], baseline[:, 6])
    with pytest.raises(Exception, match="generic D112 scorer is reserved"):
        score_d112_seam_logits(state, bank, query)


@pytest.mark.parametrize("k", [1, 5, 10])
def test_ground_head_source_held_g1_uses_fixed_ground_anchor_and_exact_m0_new(k: int) -> None:
    bundle = _source_held_g1_bundle()
    bank = _bank(k)
    state = fit_d112_ground_head_source_held_g1_state(bundle, bank)
    query, _labels = _support(k)
    actual = score_d112_seam_source_held_g1_logits(state, bank, query)
    baseline = _m0(bank, query)
    old_indices = np.asarray(state.old_class_indices)
    expected_ground = np.asarray(bundle.g, dtype=np.float64)
    expected_ground /= np.linalg.norm(expected_ground, axis=1, keepdims=True)
    expected_rho = state.v_s_amb[old_indices] / (
        state.v_s_amb[old_indices]
        + np.asarray(bundle.v_g_amb, dtype=np.float64)
        + state.discrepancy_amb[old_indices]
    )
    assert state.bundle_component_state == G1_COMPONENT_STATE
    assert state.evaluation_scope == G1_EVALUATION_SCOPE
    assert np.all(state.information_valid[old_indices])
    assert not np.any(state.donor_valid)
    assert np.allclose(state.anchors[old_indices], expected_ground, atol=1.0e-7, rtol=0.0)
    assert np.array_equal(state.alpha, np.zeros_like(state.alpha))
    assert np.array_equal(state.v_h_amb, np.zeros_like(state.v_h_amb))
    assert np.array_equal(state.anchor_shift_l2, np.zeros_like(state.anchor_shift_l2))
    assert np.allclose(state.rho[old_indices], expected_rho, atol=1.0e-7, rtol=0.0)
    assert np.max(np.abs(actual[:, :6] - baseline[:, :6])) > 0.0
    assert np.array_equal(actual[:, 6], baseline[:, 6])


def test_ground_head_source_held_g1_rejects_g0_bundle() -> None:
    with pytest.raises(Exception, match="source-held G1 fit"):
        fit_d112_ground_head_source_held_g1_state(_bundle(), _bank(1))


def test_ground_head_source_held_g1_invalid_bundle_is_exact_m0() -> None:
    bundle = _source_held_g1_bundle(global_valid=False)
    bank = _bank(1)
    query, _labels = _support(1)
    state = fit_d112_ground_head_source_held_g1_state(bundle, bank)
    assert not np.any(state.information_valid)
    assert np.array_equal(
        score_d112_seam_source_held_g1_logits(state, bank, query), _m0(bank, query)
    )


def test_source_held_g1_rejects_g0_relabel_and_cross_surface_fit() -> None:
    g0 = _bundle()
    g1 = _source_held_g1_bundle()
    relabelled = dict(g0.manifest)
    relabelled["component_state"] = G1_COMPONENT_STATE
    with pytest.raises(D112BundleError, match="source-held G1"):
        replace(g0, manifest=relabelled)
    with pytest.raises(Exception, match="source-held G1 fit"):
        fit_d112_seam_source_held_g1_state(g0, _bank(1))
    with pytest.raises(Exception, match="G0 fit"):
        fit_d112_seam_g0_state(g1, _bank(1))
    with pytest.raises(Exception, match="target formal D112 Phase2 surface"):
        fit_d112_seam_state(g1, _bank(1))


def test_source_held_g1_builder_and_fit_have_no_source_row_or_truth_surface() -> None:
    builder_parameters = set(inspect.signature(build_d112_source_held_g1_bundle).parameters)
    assert not builder_parameters & {
        "g0_bundle",
        "source_rows",
        "source_ids",
        "target",
        "query",
        "truth",
        "labels",
        "role",
        "quota",
    }
    fit_parameters = set(inspect.signature(fit_d112_seam_source_held_g1_state).parameters)
    assert fit_parameters == {"bundle", "bank"}
    ground_head_fit_parameters = set(
        inspect.signature(fit_d112_ground_head_source_held_g1_state).parameters
    )
    assert ground_head_fit_parameters == {"bundle", "bank"}
    score_parameters = set(inspect.signature(score_d112_seam_source_held_g1_logits).parameters)
    assert score_parameters == {"state", "bank", "held_query_zid"}


def test_donor_invalid_class_can_still_receive_other_class_transport() -> None:
    rows, labels = _support(1)
    ground, _q0, _basis = _geometry()
    rows[0] = -ground[0]
    bank = build_typed_zid_support_bank(rows, labels, CLASSES, config=_lock(1))
    state = fit_d112_seam_g0_state(_bundle(), bank)
    index = CLASSES.index("old-0")
    assert not state.donor_valid[index]
    assert state.information_valid[index]
    assert state.rho[index] > 0.0
    assert state.anchor_shift_l2[index] > 0.0


def test_recipient_own_support_does_not_change_its_loo_anchor() -> None:
    rows, labels = _support(1)
    first = build_typed_zid_support_bank(rows, labels, CLASSES, config=_lock(1))
    changed = rows.copy()
    changed[0, 50] = 0.2
    changed[0] /= np.linalg.norm(changed[0])
    second = build_typed_zid_support_bank(changed, labels, CLASSES, config=_lock(1))
    first_state = fit_d112_seam_g0_state(_bundle(), first)
    second_state = fit_d112_seam_g0_state(_bundle(), second)
    assert np.array_equal(first_state.anchors[0], second_state.anchors[0])
