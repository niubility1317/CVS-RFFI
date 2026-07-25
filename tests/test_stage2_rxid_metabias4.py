from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_bundle import (
    RXIDMetaBias4BundleError,
    build_rxid_metabias4_bundle,
    deserialize_rxid_metabias4_bundle,
    serialize_rxid_metabias4_bundle,
)
from cvsrffi.stage2_rb_metabias4_qknn import (
    Phase1MetaBias4Lock,
    baseline_zid_from_pre_relu,
    build_d102_baseline_bank,
    build_phase1_metabias4_asset,
    fit_d102_stage2_state,
)
from cvsrffi.stage2_rxid_metabias4 import (
    ACTIVE,
    INACTIVE_NON_PROMOTABLE,
    K1IdentifiabilityReceipt,
    RXIDMetaBias4Stage2Error,
    audit_d103_int8,
    audit_d103_resources,
    fit_d103_stage2_state,
    predict_d103_class_indices,
    predict_d103_logits,
    serialize_d103_runtime_state,
    solve_d103_support_coefficient,
    stable_first_argmax,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    score_zid_student_t_logits,
)


CLASSES = ("opaque-2", "opaque-0", "opaque-1")
HASHES = tuple(f"{index:x}" * 64 for index in range(1, 9))


def _bundle():
    rng = np.random.default_rng(103713)
    u = np.zeros((32, 160), dtype=np.float32)
    u[:, :32] = np.eye(32, dtype=np.float32)
    b = rng.normal(0.0, 0.08, (160, 4)).astype(np.float32)
    g = np.zeros((5, 32), dtype=np.float32)
    for index in range(5):
        g[index, index] = 1.0
        g[index, (index + 7) % 32] = 0.3
    t = np.asarray(
        [
            [0.15, 0.08, -0.05, 0.10],
            [0.12, 0.06, -0.03, 0.08],
            [0.10, 0.05, -0.02, 0.07],
            [0.14, 0.07, -0.04, 0.09],
            [0.11, 0.04, -0.01, 0.06],
        ],
        dtype=np.float32,
    )
    return build_rxid_metabias4_bundle(
        u,
        b,
        g,
        t,
        np.full((5, 4), 4.0, dtype=np.float32),
        np.full(5, 1.8, dtype=np.float32),
        cell_min_physical_count=np.full(5, 2, dtype=np.int16),
        cell_class_count=np.full(5, 3, dtype=np.int16),
        checkpoint_sha256=HASHES[0],
        runtime_sha256=HASHES[1],
        method_lock_sha256=HASHES[2],
        training_receipt_sha256=HASHES[3],
        nested_receipt_sha256=HASHES[4],
        tx_probe_receipt_sha256=HASHES[5],
        aggregation_receipt_sha256=HASHES[6],
        quantization_receipt_sha256=HASHES[7],
        tx_probe_mean_balanced_accuracy=0.20,
        tx_probe_max_balanced_accuracy=0.24,
    )


def _qknn(k: int, *, h0: float = 0.2) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=3.0,
        kernel_effective_dim=160,
        kernel_volume_gamma=1.0,
        shared_h0=h0,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="a" * 64,
        quantization_margin_audit_sha256="b" * 64,
    )


def _support(k: int):
    rng = np.random.default_rng(607 + k)
    pre, zdom, labels = [], [], []
    for class_index, label in enumerate(CLASSES):
        for shot in range(k):
            row = rng.normal(0.1, 0.15, 160).astype(np.float32)
            row[20 + class_index] += np.float32(1.0)
            domain = rng.normal(0.0, 0.03, 160).astype(np.float32)
            domain[class_index] += np.float32(1.0)
            pre.append(row)
            zdom.append(domain)
            labels.append(label)
    return (
        np.asarray(pre, dtype=np.float32),
        np.asarray(zdom, dtype=np.float32),
        np.asarray(labels, dtype=str),
    )


def _k1_receipt(**overrides):
    values = {
        "view_top1_agreement": 1.0,
        "large_margin_flip_count": 0,
        "independent_direction_cosine_median": 0.9,
        "independent_episode_count": 7,
        "receipt_sha256": "c" * 64,
    }
    values.update(overrides)
    return K1IdentifiabilityReceipt(**values)


def _fit(k: int, *, receipt=None):
    pre, zdom, labels = _support(k)
    state = fit_d103_stage2_state(
        _bundle(),
        pre,
        zdom,
        labels,
        CLASSES,
        qknn_config=_qknn(k),
        stage="S_C",
        support_receipt_sha256="d" * 64,
        k1_identifiability_receipt=receipt,
    )
    return state, pre, zdom, labels


def test_k1_active_reuses_d102_closed_form_and_typed_qknn() -> None:
    state, pre, _, _ = _fit(1, receipt=_k1_receipt())
    assert state.status == ACTIVE
    assert state.fit_audit["d102_closed_form_reused"] is True
    assert state.fit_audit["typed_qknn_reused"] is True
    assert all(state.fit_audit["k1_numeric_gates"].values())
    assert all(state.fit_audit["k1_external_gates"].values())
    assert state.fit_audit["target25_authorized"] is False
    logits = predict_d103_logits(state, pre)
    assert logits.shape == (len(pre), len(CLASSES))
    assert tuple(state.bank.classes) == CLASSES
    resources = audit_d103_resources(state)
    int8 = audit_d103_int8(state, pre, _support(1)[2], pre)
    assert int8["top1_agreement"] >= 0.995
    assert int8["large_margin_flip_count"] == 0
    assert int8["passes_d103_int8_gate"] is True
    assert resources["fp32_persistent_sidecar_bytes"] == 0
    assert resources["fp16_persistent_learned_sidecar_bytes"] == 0
    assert resources["trainable_parameters_stage2"] == 0
    assert resources["optimizer_steps_stage2"] == 0


def test_k1_missing_or_failed_gate_is_prediction_complete_exact_m0() -> None:
    for receipt in (
        None,
        _k1_receipt(view_top1_agreement=0.994),
        _k1_receipt(large_margin_flip_count=1),
        _k1_receipt(independent_direction_cosine_median=0.79),
    ):
        state, pre, _, labels = _fit(1, receipt=receipt)
        assert state.status == INACTIVE_NON_PROMOTABLE
        assert state.active is False
        assert state.fit_audit["d103_instance_rejected"] is True
        assert state.fit_audit["inactive_fold_counts_as_success"] is False
        assert state.fit_audit["target25_authorized"] is False
        baseline_bank, metric, _ = build_d102_baseline_bank(
            pre, labels, CLASSES, qknn_config=_qknn(1)
        )
        expected = score_zid_student_t_logits(
            baseline_bank, baseline_zid_from_pre_relu(pre), metric=metric
        )
        np.testing.assert_array_equal(predict_d103_logits(state, pre), expected)
        assert predict_d103_logits(state, pre).shape == (
            len(pre),
            len(CLASSES),
        )


def test_query_is_read_only_all_class_and_registry_ties_are_stable() -> None:
    state, pre, _, _ = _fit(5)
    before = serialize_d103_runtime_state(state)
    together = predict_d103_logits(state, pre[:8])
    chunked = np.concatenate(
        (
            predict_d103_logits(state, pre[:3]),
            predict_d103_logits(state, pre[3:8]),
        )
    )
    repeated = predict_d103_logits(state, pre[:8])
    after = serialize_d103_runtime_state(state)
    np.testing.assert_array_equal(together, chunked)
    np.testing.assert_array_equal(together, repeated)
    assert before == after
    assert together.shape[1] == len(CLASSES)
    np.testing.assert_array_equal(
        stable_first_argmax(
            np.asarray([[1.0, 1.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32),
            axis=1,
        ),
        np.asarray([0, 0]),
    )
    assert predict_d103_class_indices(state, pre[:8]).shape == (8,)
    with pytest.raises(TypeError):
        predict_d103_logits(state, pre[:1], query_truth=["opaque-2"])
    forbidden = {
        "query_labels",
        "query_truth",
        "receiver",
        "tx",
        "old_roles",
        "new_roles",
        "class_quota",
    }
    assert not set(inspect.signature(fit_d103_stage2_state).parameters).intersection(
        forbidden
    )
    assert not set(inspect.signature(predict_d103_logits).parameters).intersection(
        forbidden
    )


def test_frozen_qknn_constants_are_not_tunable() -> None:
    pre, zdom, labels = _support(5)
    with pytest.raises(RXIDMetaBias4Stage2Error, match="constants drift"):
        fit_d103_stage2_state(
            _bundle(),
            pre,
            zdom,
            labels,
            CLASSES,
            qknn_config=_qknn(5, h0=0.2001),
            stage="S_C",
            support_receipt_sha256="d" * 64,
        )


def test_bundle_wire_restore_preserves_stage2_predictions() -> None:
    pre, zdom, labels = _support(5)
    original = _bundle()
    restored = deserialize_rxid_metabias4_bundle(
        serialize_rxid_metabias4_bundle(original)
    )
    kwargs = {
        "qknn_config": _qknn(5),
        "stage": "S_C",
        "support_receipt_sha256": "d" * 64,
    }
    first = fit_d103_stage2_state(
        original, pre, zdom, labels, CLASSES, **kwargs
    )
    second = fit_d103_stage2_state(
        restored, pre, zdom, labels, CLASSES, **kwargs
    )
    np.testing.assert_array_equal(first.coefficient_fp16, second.coefficient_fp16)
    np.testing.assert_array_equal(
        predict_d103_logits(first, pre), predict_d103_logits(second, pre)
    )
    assert first.state_receipt_sha256 == second.state_receipt_sha256


def test_failed_tx_probe_remains_prediction_complete_but_not_deployable() -> None:
    failed = replace(
        _bundle(),
        tx_probe_mean_balanced_accuracy=0.31,
        tx_probe_max_balanced_accuracy=0.37,
        content_root_sha256="",
    )
    pre, zdom, labels = _support(5)
    solution = solve_d103_support_coefficient(
        failed, zdom, labels, CLASSES, active_k=5
    )
    assert solution.fit_audit["tx_probe_gate_pass"] is False
    state = fit_d103_stage2_state(
        failed,
        pre,
        zdom,
        labels,
        CLASSES,
        qknn_config=_qknn(5),
        stage="S_C",
        support_receipt_sha256="d" * 64,
    )
    assert predict_d103_class_indices(state, pre).shape == (len(pre),)
    resources = audit_d103_resources(state)
    assert resources["actual_serialized_state_bytes"] is None
    assert resources["passes_state_gate"] is False
    assert resources["deployment_serialization_authorized"] is False
    with pytest.raises(RXIDMetaBias4BundleError, match="diagnostic-only"):
        serialize_d103_runtime_state(state)


def test_support_solver_is_fp16_identical_to_reviewed_d102_closed_form() -> None:
    bundle = _bundle()
    pre, zdom, labels = _support(5)
    lock = Phase1MetaBias4Lock(
        checkpoint_sha256=bundle.checkpoint_sha256,
        runtime_sha256=bundle.runtime_sha256,
        bundle_sha256=bundle.content_root_sha256,
        external_seal_sha256=bundle.content_root_sha256,
        method_lock_sha256=bundle.method_lock_sha256,
        receiver_held_receipt_sha256=bundle.training_receipt_sha256,
        class_loco_receipt_sha256=bundle.nested_receipt_sha256,
        tx_probe_receipt_sha256=bundle.tx_probe_receipt_sha256,
        label_permutation_receipt_sha256=bundle.method_lock_sha256,
        aggregation_receipt_sha256=bundle.aggregation_receipt_sha256,
        quantization_receipt_sha256=bundle.quantization_receipt_sha256,
        tx_probe_balanced_accuracy=bundle.tx_probe_max_balanced_accuracy,
    )
    asset = build_phase1_metabias4_asset(
        bundle.decode_b().astype(np.float32),
        bundle.decode_u().astype(np.float32),
        bundle.decode_bank_g().astype(np.float32),
        bundle.decode_bank_t().astype(np.float32),
        bundle.decode_bank_precision().astype(np.float32),
        bundle.decode_bank_sigma().astype(np.float32),
        bundle.lambda0_fp16.astype(np.float32),
        bundle.amax_fp16.astype(np.float32),
        temperature=float(bundle.temperature_fp16),
        ellipsoid_radius=float(bundle.radius_fp16),
        cell_min_physical_count=bundle.cell_min_physical_count_int16,
        cell_class_count=bundle.cell_class_count_int16,
        lock=lock,
    )
    reviewed = fit_d102_stage2_state(
        asset,
        pre,
        zdom,
        labels,
        CLASSES,
        qknn_config=_qknn(5),
        stage="S_C",
        support_receipt_sha256="d" * 64,
    )
    direct = solve_d103_support_coefficient(
        bundle, zdom, labels, CLASSES, active_k=5
    )
    np.testing.assert_array_equal(direct.coefficient_fp16, reviewed.a_fp16)
