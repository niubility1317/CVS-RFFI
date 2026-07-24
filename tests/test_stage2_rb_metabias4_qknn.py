from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

from cvsrffi.stage2_rb_metabias4_qknn import (
    CODE_DIM,
    DOMAIN_DIM,
    MAX_POST_BACKBONE_MAC_PER_QUERY,
    MAX_STATE_BYTES,
    Phase1MetaBias4Lock,
    RBMetaBias4Error,
    audit_d102_int8,
    audit_d102_query_geometry,
    audit_d102_resources,
    audit_d102_stage_lifecycle,
    build_d102_baseline_bank,
    build_phase1_metabias4_asset,
    build_phase1_metabias4_asset_from_bundle,
    decode_metabias_basis,
    fit_d102_stage2_state,
    predict_d102_logits,
    serialize_d102_runtime_state,
    transform_d102_query,
)
from cvsrffi.phase1_rb_metabias4_bundle import (
    RBMetaBias4Config,
    build_phase1_rb_metabias4_bundle,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    score_zid_student_t_logits,
)


CLASSES = ("opaque-a", "opaque-b", "opaque-c")


def _qknn_lock(k: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _phase1_lock(
    tx_probe: float = 0.20, **overrides: object
) -> Phase1MetaBias4Lock:
    values = [f"{index:x}" * 64 for index in range(1, 12)]
    fields: dict[str, object] = {
        "checkpoint_sha256": values[0],
        "runtime_sha256": values[1],
        "bundle_sha256": values[2],
        "external_seal_sha256": values[3],
        "method_lock_sha256": values[4],
        "receiver_held_receipt_sha256": values[5],
        "class_loco_receipt_sha256": values[6],
        "tx_probe_receipt_sha256": values[7],
        "label_permutation_receipt_sha256": values[8],
        "aggregation_receipt_sha256": values[9],
        "quantization_receipt_sha256": values[10],
        "tx_probe_balanced_accuracy": tx_probe,
    }
    fields.update(overrides)
    return Phase1MetaBias4Lock(**fields)


def _receipt_digest(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _asset(
    *,
    t_scale: float = 0.22,
    a_max: float = 0.75,
    radius: float = 1.0,
    physical_min: int = 2,
):
    rng = np.random.default_rng(102)
    basis = np.zeros((160, CODE_DIM), dtype=np.float32)
    for column in range(CODE_DIM):
        basis[column, column] = np.float32(0.32 + 0.04 * column)
        basis[8 + column, column] = np.float32(-0.25)
        basis[40 + column, column] = np.float32(0.18)
    domain_u = np.zeros((DOMAIN_DIM, 160), dtype=np.float32)
    domain_u[:, :DOMAIN_DIM] = np.eye(DOMAIN_DIM, dtype=np.float32)
    domain_u += np.float32(0.002) * rng.normal(
        size=domain_u.shape
    ).astype(np.float32)
    bank_g = rng.normal(size=(5, DOMAIN_DIM)).astype(np.float32)
    bank_t = (
        t_scale
        * np.asarray(
            [
                [1.0, -0.4, 0.2, 0.6],
                [0.5, 0.8, -0.7, 0.2],
                [-0.6, 0.4, 0.9, -0.3],
                [0.3, -0.9, 0.4, 0.8],
                [-0.2, 0.5, -0.4, 1.0],
            ],
            dtype=np.float32,
        )
    )
    precision = np.asarray(
        [
            [1.0, 1.2, 0.8, 1.1],
            [0.9, 1.0, 1.3, 0.7],
            [1.2, 0.8, 1.0, 1.1],
            [0.8, 1.3, 0.9, 1.0],
            [1.1, 0.9, 1.2, 0.8],
        ],
        dtype=np.float32,
    )
    return build_phase1_metabias4_asset(
        basis,
        domain_u,
        bank_g,
        bank_t,
        precision,
        np.full(5, 0.9, dtype=np.float32),
        np.asarray([0.7, 0.9, 1.1, 1.3], dtype=np.float32),
        np.full(CODE_DIM, a_max, dtype=np.float32),
        temperature=0.55,
        ellipsoid_radius=radius,
        cell_min_physical_count=np.full(5, physical_min, dtype=np.int16),
        cell_class_count=np.full(5, 3, dtype=np.int16),
        lock=_phase1_lock(),
    )


def _support(k: int, classes=CLASSES):
    rng = np.random.default_rng(607 + k + len(classes))
    pre = []
    zdom = []
    labels = []
    for class_index, label in enumerate(classes):
        for shot in range(k):
            row = rng.normal(0.0, 0.16, 160).astype(np.float32)
            row[20 + class_index] += np.float32(1.2)
            row[class_index] = np.float32(-0.015 + 0.004 * shot)
            domain = rng.normal(0.0, 0.2, 160).astype(np.float32)
            domain[class_index * 3 : class_index * 3 + 3] += np.float32(0.9)
            pre.append(row)
            zdom.append(domain)
            labels.append(label)
    return (
        np.asarray(pre, dtype=np.float32),
        np.asarray(zdom, dtype=np.float32),
        np.asarray(labels, dtype=str),
    )


def _fit(k: int = 5, *, stage: str = "S_C", classes=CLASSES, asset=None):
    pre, zdom, labels = _support(k, classes)
    state = fit_d102_stage2_state(
        _asset() if asset is None else asset,
        pre,
        zdom,
        labels,
        classes,
        qknn_config=_qknn_lock(k),
        stage=stage,
        support_receipt_sha256="d" * 64,
    )
    return state, pre, zdom, labels


def test_unique_diagonal_solution_and_fixed_box_then_ellipsoid_map() -> None:
    asset = _asset(t_scale=8.0, a_max=0.08, radius=0.04)
    state, _, _, _ = _fit(k=1, asset=asset)
    audit = state.fit_audit
    a_tilde = np.asarray(audit["a_tilde"])
    a_box = np.asarray(audit["a_box"])
    a_mapped = np.asarray(audit["a_mapped"])
    limits = asset.a_max_fp16.astype(np.float64)
    lambda0 = asset.lambda0_fp16.astype(np.float64)

    np.testing.assert_allclose(
        a_box, np.clip(a_tilde, -limits, limits), rtol=0.0, atol=0.0
    )
    expected = (
        float(asset.ellipsoid_radius_fp16)
        * a_box
        / np.sqrt(np.sum(lambda0 * a_box * a_box))
    )
    np.testing.assert_allclose(a_mapped, expected, rtol=0.0, atol=1.0e-12)
    assert audit["box_constraint_active"] is True
    assert audit["ellipsoid_constraint_active"] is True
    assert audit["data_information_rank"] == 4
    assert 0.0 < audit["prior_fraction"] < 1.0
    assert audit["singleton_per_class"] is True


def test_s_b_s_c_use_same_class_equal_formula_and_class_permutation() -> None:
    asset = _asset()
    old_classes = CLASSES[:2]
    old_pre, old_zdom, old_labels = _support(5, old_classes)
    state_b = fit_d102_stage2_state(
        asset,
        old_pre,
        old_zdom,
        old_labels,
        old_classes,
        qknn_config=_qknn_lock(5),
        stage="S_B",
        support_receipt_sha256="c" * 64,
    )
    pre, zdom, labels = _support(5, CLASSES)
    state_c = fit_d102_stage2_state(
        asset,
        pre,
        zdom,
        labels,
        CLASSES,
        qknn_config=_qknn_lock(5),
        stage="S_C",
        support_receipt_sha256="d" * 64,
    )
    permuted = tuple(reversed(CLASSES))
    state_c_permuted = fit_d102_stage2_state(
        asset,
        pre,
        zdom,
        labels,
        permuted,
        qknn_config=state_b.bank.config,
        stage="S_C",
        support_receipt_sha256="d" * 64,
    )
    np.testing.assert_array_equal(state_c_permuted.a_fp16, state_c.a_fp16)
    assert state_b.fit_audit["per_class_weights"] == [0.5, 0.5]
    assert state_c.fit_audit["per_class_weights"] == [1.0 / 3.0] * 3
    assert state_c.fit_audit["old_new_role_weights_present"] is False
    assert state_c.fit_audit["new_class_count_weight_present"] is False
    lifecycle = audit_d102_stage_lifecycle(state_b, state_c)
    assert lifecycle["s_c_reencodes_all_registered_support"] is True
    assert lifecycle["s_c_reuses_s_b_bank_prefix"] is False
    assert lifecycle["total_support_state_build_mac"] > 0


def test_pre_relu_bias_relu_norm_and_typed_qknn_are_reused() -> None:
    state, pre, _, labels = _fit()
    transformed = transform_d102_query(state, pre)
    np.testing.assert_allclose(
        np.linalg.norm(transformed, axis=1), 1.0, rtol=0.0, atol=2.0e-6
    )
    assert state.support_geometry_audit["relu_mask_change_count"] > 0
    assert state.support_geometry_audit["non_common_geometry_change"] is True
    logits = predict_d102_logits(state, pre)
    direct = score_zid_student_t_logits(
        state.bank, transformed, metric=state.metric
    )
    np.testing.assert_array_equal(logits, direct)
    assert logits.shape == (len(pre), len(CLASSES))
    assert tuple(state.bank.classes) == CLASSES
    assert labels.shape[0] == state.bank.support_row_count


def test_query_is_read_only_truth_free_all_class_and_chunk_equivalent() -> None:
    state, pre, _, labels = _fit()
    baseline_bank, _, _ = build_d102_baseline_bank(
        pre, labels, CLASSES, qknn_config=state.bank.config
    )
    query = pre[:7].copy()
    before = serialize_d102_runtime_state(state)
    together = predict_d102_logits(state, query)
    chunked = np.concatenate(
        [
            predict_d102_logits(state, query[:2]),
            predict_d102_logits(state, query[2:]),
        ]
    )
    repeated = predict_d102_logits(state, query)
    after = serialize_d102_runtime_state(state)
    np.testing.assert_array_equal(together, chunked)
    np.testing.assert_array_equal(together, repeated)
    assert before == after
    assert together.shape[1] == len(CLASSES)
    audit = audit_d102_query_geometry(state, baseline_bank, query)
    assert audit["state_unchanged_after_query"] is True
    assert audit["query_truth_read"] is False
    assert audit["all_registered_classes_compete"] is True
    with pytest.raises(TypeError):
        predict_d102_logits(state, query, query_truth=labels[:7])

    fit_names = set(inspect.signature(fit_d102_stage2_state).parameters)
    predict_names = set(inspect.signature(predict_d102_logits).parameters)
    forbidden = {
        "query",
        "query_labels",
        "query_truth",
        "receiver",
        "tx",
        "old_roles",
        "new_roles",
        "class_quota",
    }
    assert not fit_names.intersection(forbidden)
    assert not predict_names.intersection(forbidden - {"query"})


def test_int8_state_and_mac_audits_close_without_fp32_sidecar() -> None:
    state, pre, _, labels = _fit()
    int8 = audit_d102_int8(state, pre, labels, pre[:9])
    resources = audit_d102_resources(state)
    assert int8["top1_agreement"] >= 0.995
    assert int8["large_margin_flip_count"] == 0
    assert int8["passes_d102_int8_gate"] is True
    assert resources["actual_serialized_state_bytes"] <= MAX_STATE_BYTES
    assert (
        resources["post_backbone_mac_per_query"]
        <= MAX_POST_BACKBONE_MAC_PER_QUERY
    )
    assert resources["fp32_persistent_sidecar_bytes"] == 0
    assert resources["trainable_parameters_stage2"] == 0
    assert resources["optimizer_steps_stage2"] == 0
    assert resources["bank_matching_mac_support"] > 0
    assert resources["metabias_support_reencode_mac"] == 160 * 4
    assert resources["passes_state_gate"] is True
    assert resources["passes_query_mac_gate"] is True


def test_phase1_asset_rejects_tx_leak_and_single_physical_aggregate() -> None:
    with pytest.raises(RBMetaBias4Error, match="exceeds 25%"):
        _phase1_lock(tx_probe=0.2501)
    with pytest.raises(RBMetaBias4Error, match=">=2 physical"):
        _asset(physical_min=1)


def test_fit_rejects_unbalanced_support_and_any_query_like_kwarg() -> None:
    asset = _asset()
    pre, zdom, labels = _support(1)
    with pytest.raises(RBMetaBias4Error, match="balanced K-shot"):
        fit_d102_stage2_state(
            asset,
            pre[:-1],
            zdom[:-1],
            labels[:-1],
            CLASSES,
            qknn_config=_qknn_lock(1),
            stage="S_C",
            support_receipt_sha256="d" * 64,
        )
    with pytest.raises(TypeError):
        fit_d102_stage2_state(
            asset,
            pre,
            zdom,
            labels,
            CLASSES,
            qknn_config=_qknn_lock(1),
            stage="S_C",
            support_receipt_sha256="d" * 64,
            query_pre_relu=pre,
        )


def test_verified_phase1_bundle_factory_closes_parallel_wire_formats() -> None:
    rng = np.random.default_rng(607102)
    classes = np.asarray(["c0", "c1", "c2"], dtype=np.str_)
    pre_relu: list[np.ndarray] = []
    z_dom: list[np.ndarray] = []
    labels: list[str] = []
    receivers: list[str] = []
    days: list[str] = []
    physical: list[str] = []
    for receiver_index in range(3):
        for day_index in range(2):
            for class_index, class_id in enumerate(classes):
                for sample_index in range(3):
                    pre_relu.append(
                        rng.normal(size=160)
                        + 0.2 * receiver_index
                        + 0.1 * class_index
                    )
                    z_dom.append(
                        rng.normal(size=160)
                        + 0.3 * receiver_index
                        - 0.1 * class_index
                    )
                    labels.append(str(class_id))
                    receivers.append(f"r{receiver_index}")
                    days.append(f"d{day_index}")
                    physical.append(
                        f"p{receiver_index}-{day_index}-{class_index}-{sample_index}"
                    )
    bundle = build_phase1_rb_metabias4_bundle(
        {
            "pre_relu": np.asarray(pre_relu, dtype=np.float32),
            "z_dom": np.asarray(z_dom, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.str_),
            "receiver_ids": np.asarray(receivers, dtype=np.str_),
            "day_ids": np.asarray(days, dtype=np.str_),
            "physical_ids": np.asarray(physical, dtype=np.str_),
            "class_ids": classes,
        },
        checkpoint_sha256="1" * 64,
        runtime_sha256="2" * 64,
        method_lock_sha256="5" * 64,
        config=RBMetaBias4Config(),
    )
    lock = _phase1_lock(
        checkpoint_sha256=bundle.checkpoint_sha256,
        runtime_sha256=bundle.runtime_sha256,
        bundle_sha256=bundle.content_root_sha256,
        method_lock_sha256=bundle.method_lock_sha256,
        aggregation_receipt_sha256=_receipt_digest(
            dict(bundle.aggregation_receipt)
        ),
        quantization_receipt_sha256=_receipt_digest(
            dict(bundle.quantization_receipt)
        ),
    )
    asset = build_phase1_metabias4_asset_from_bundle(bundle, lock=lock)
    assert asset.bank_cell_count == bundle.bank_count
    assert decode_metabias_basis(asset).shape == (160, 4)
    assert np.all(asset.cell_min_physical_count_int16 >= 2)
    assert np.all(asset.cell_class_count_int16 >= 2)
