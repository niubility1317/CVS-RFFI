from __future__ import annotations

import inspect
import hashlib
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from cvsrffi.stage2_lpo_rc_qknn import (
    LPO_RC_RECEIPT_SCHEMA,
    LPO_RC_SCHEMA,
    LPORCQKNNError,
    LPORCProtocolHandleError,
    TypedValidatedOnceP2SplitHandle,
    audit_lpo_rc_resource,
    build_lpo_rc_qknn_state,
    score_lpo_rc_qknn_logits,
    serialize_lpo_rc_qknn_state,
    validate_lpo_rc_physical_id_disjointness,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Z_DIM,
    Phase1ZIDStudentTLock,
    TypedMetricProvenanceReceipt,
    _canonical_sha256,
    build_typed_shared_psd_metric,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


CLASSES = ("opaque_a", "opaque_b", "opaque_c", "opaque_d")


def _lock(k_shot: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
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


def _support(
    k_shot: int,
    seed: int = 105731,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    centers = normalize_zid_rows(
        rng.normal(size=(len(CLASSES), Z_DIM)).astype(np.float32)
    )
    rows: list[np.ndarray] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for class_index, class_id in enumerate(CLASSES):
        noise = 0.015 + 0.01 * class_index
        rows.append(
            (
                centers[class_index]
                + noise * rng.normal(size=(k_shot, Z_DIM))
            ).astype(np.float32)
        )
        labels.extend([class_id] * k_shot)
        physical_ids.extend(
            [f"physical-{class_index}-{support_index}" for support_index in range(k_shot)]
        )
    return np.concatenate(rows), tuple(labels), tuple(physical_ids)


def _build(k_shot: int):
    support, labels, physical_ids = _support(k_shot)
    config = _lock(k_shot)
    bank = build_typed_zid_support_bank(
        support,
        labels,
        CLASSES,
        config=config,
    )
    metric = identity_shared_psd_metric(config=config)
    handle = _split_handle(physical_ids)
    state = build_lpo_rc_qknn_state(
        bank,
        support,
        labels,
        CLASSES,
        metric=metric,
        support_physical_ids=physical_ids,
        split_handle=handle,
    )
    return support, labels, physical_ids, bank, metric, handle, state


def _split_handle(
    physical_ids: tuple[str, ...],
    *,
    capsule: str = "3",
    split: str = "4",
) -> TypedValidatedOnceP2SplitHandle:
    return TypedValidatedOnceP2SplitHandle(
        capsule_id=capsule * 64,
        split_id=split * 64,
        validator_receipt_sha256="5" * 64,
        support_physical_root_sha256=_canonical_sha256(sorted(physical_ids)),
        query_physical_root_sha256="6" * 64,
        support_query_disjoint=True,
    )


def _rank1_metric(config: Phase1ZIDStudentTLock):
    basis = np.zeros((1, Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    provenance = TypedMetricProvenanceReceipt(
        fit_scope="target_support_only",
        source_receipt_sha256="7" * 64,
        query_rows_used_for_fit=0,
    )
    return build_typed_shared_psd_metric(
        basis,
        np.asarray([0.125], dtype=np.float32),
        config=config,
        source="lpo_rc_rank1_test",
        provenance=provenance,
    )


def _queries(seed: int = 105732) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(7, Z_DIM)).astype(np.float32)


def test_k1_is_exact_base_identity_and_has_no_loo_work() -> None:
    _, _, _, bank, metric, handle, state = _build(1)
    query = _queries()
    base = score_zid_student_t_logits(bank, query, metric=metric)
    head = score_lpo_rc_qknn_logits(
        state, query, bank=bank, metric=metric, split_handle=handle
    )
    np.testing.assert_array_equal(head, base)
    assert head.tobytes(order="C") == base.tobytes(order="C")
    np.testing.assert_array_equal(state.bias_fp16, np.zeros(len(CLASSES), dtype=np.float16))
    receipt = audit_lpo_rc_resource(state, bank, metric, handle)
    assert receipt["physical_loo_enabled"] is False
    assert receipt["loo_fit_total_matmul_mac"] == 0
    assert receipt["query_bias_add_ops"] == len(CLASSES)


@pytest.mark.parametrize("k_shot", [5, 10])
def test_lpo_rc_preserves_bank_scales_and_adds_only_fixed_class_bias(
    k_shot: int,
) -> None:
    _, _, _, bank, metric, handle, state = _build(k_shot)
    query = _queries()
    base = score_zid_student_t_logits(bank, query, metric=metric)
    head = score_lpo_rc_qknn_logits(
        state, query, bank=bank, metric=metric, split_handle=handle
    )
    np.testing.assert_array_equal(state.class_scales_fp16, bank.class_scales_fp16)
    assert np.any(state.bias_fp16 != np.float16(0.0))
    np.testing.assert_allclose(
        head.astype(np.float64) - base.astype(np.float64),
        np.broadcast_to(state.bias_fp16.astype(np.float64), head.shape),
        rtol=0.0,
        atol=2.0e-7,
    )
    receipt = audit_lpo_rc_resource(state, bank, metric, handle)
    assert receipt["physical_loo_enabled"] is True
    assert receipt["physical_loo_self_exclusion"] is True
    assert receipt["loo_fit_score_kernel_evaluations"] == (
        len(CLASSES) * k_shot * (len(CLASSES) * k_shot - 1)
    )
    assert receipt["query_extra_dot_product_MAC"] == 0
    assert receipt["query_bias_add_ops"] == len(CLASSES)
    assert abs(receipt["bias_sum_fp64"]) < 2.0e-3


@pytest.mark.parametrize("k_shot", [1, 5, 10])
def test_query_order_chunking_and_state_are_invariant(k_shot: int) -> None:
    _, _, _, bank, metric, handle, state = _build(k_shot)
    query = _queries()
    receipt_hash = state.receipt_sha256
    full = score_lpo_rc_qknn_logits(
        state, query, bank=bank, metric=metric, split_handle=handle
    )
    chunked = np.concatenate(
        [
            score_lpo_rc_qknn_logits(
                state, query[:2], bank=bank, metric=metric, split_handle=handle
            ),
            score_lpo_rc_qknn_logits(
                state, query[2:], bank=bank, metric=metric, split_handle=handle
            ),
        ]
    )
    reordered = score_lpo_rc_qknn_logits(
        state,
        query[::-1],
        bank=bank,
        metric=metric,
        split_handle=handle,
    )[::-1]
    np.testing.assert_array_equal(chunked, full)
    np.testing.assert_array_equal(reordered, full)
    assert state.receipt_sha256 == receipt_hash
    assert state.receipt["query_state_updates"] == 0
    assert state.receipt["query_batch_dependency"] is False


@pytest.mark.parametrize("k_shot", [1, 5])
def test_class_registry_permutation_is_equivariant(k_shot: int) -> None:
    support, labels, physical_ids = _support(k_shot)
    config = _lock(k_shot)
    metric = identity_shared_psd_metric(config=config)
    handle = _split_handle(physical_ids)
    first_bank = build_typed_zid_support_bank(
        support,
        labels,
        CLASSES,
        config=config,
    )
    first_state = build_lpo_rc_qknn_state(
        first_bank,
        support,
        labels,
        CLASSES,
        metric=metric,
        support_physical_ids=physical_ids,
        split_handle=handle,
    )
    permuted = ("opaque_c", "opaque_a", "opaque_d", "opaque_b")
    second_bank = build_typed_zid_support_bank(
        support,
        labels,
        permuted,
        config=config,
    )
    second_state = build_lpo_rc_qknn_state(
        second_bank,
        support,
        labels,
        permuted,
        metric=metric,
        support_physical_ids=physical_ids,
        split_handle=handle,
    )
    for class_id in CLASSES:
        first_index = first_state.classes.index(class_id)
        second_index = second_state.classes.index(class_id)
        assert first_state.bias_fp16[first_index] == second_state.bias_fp16[second_index]
    query = _queries()
    first_logits = score_lpo_rc_qknn_logits(
        first_state,
        query,
        bank=first_bank,
        metric=metric,
        split_handle=handle,
    )
    second_logits = score_lpo_rc_qknn_logits(
        second_state,
        query,
        bank=second_bank,
        metric=metric,
        split_handle=handle,
    )
    for class_id in CLASSES:
        np.testing.assert_array_equal(
            first_logits[:, first_state.classes.index(class_id)],
            second_logits[:, second_state.classes.index(class_id)],
        )


def test_rank1_class_registry_permutation_is_equivariant() -> None:
    support, labels, physical_ids = _support(5)
    config = _lock(5)
    metric = _rank1_metric(config)
    handle = _split_handle(physical_ids)
    first_bank = build_typed_zid_support_bank(
        support,
        labels,
        CLASSES,
        config=config,
    )
    first_state = build_lpo_rc_qknn_state(
        first_bank,
        support,
        labels,
        CLASSES,
        metric=metric,
        support_physical_ids=physical_ids,
        split_handle=handle,
    )
    permuted = ("opaque_c", "opaque_a", "opaque_d", "opaque_b")
    second_bank = build_typed_zid_support_bank(
        support,
        labels,
        permuted,
        config=config,
    )
    second_state = build_lpo_rc_qknn_state(
        second_bank,
        support,
        labels,
        permuted,
        metric=metric,
        support_physical_ids=physical_ids,
        split_handle=handle,
    )
    for class_id in CLASSES:
        first_index = first_state.classes.index(class_id)
        second_index = second_state.classes.index(class_id)
        assert first_state.bias_fp16[first_index] == second_state.bias_fp16[second_index]
    query = _queries()
    first_logits = score_lpo_rc_qknn_logits(
        first_state,
        query,
        bank=first_bank,
        metric=metric,
        split_handle=handle,
    )
    second_logits = score_lpo_rc_qknn_logits(
        second_state,
        query,
        bank=second_bank,
        metric=metric,
        split_handle=handle,
    )
    for class_id in CLASSES:
        np.testing.assert_array_equal(
            first_logits[:, first_state.classes.index(class_id)],
            second_logits[:, second_state.classes.index(class_id)],
        )


def test_physical_id_contract_rejects_duplicates_and_query_overlap() -> None:
    support, labels, physical_ids, bank, metric, handle, _ = _build(5)
    duplicate_ids = list(physical_ids)
    duplicate_ids[-1] = duplicate_ids[0]
    with pytest.raises(LPORCQKNNError, match="unique"):
        build_lpo_rc_qknn_state(
            bank,
            support,
            labels,
            CLASSES,
            metric=metric,
            support_physical_ids=duplicate_ids,
            split_handle=handle,
        )
    with pytest.raises(LPORCQKNNError, match="disjoint"):
        validate_lpo_rc_physical_id_disjointness(
            physical_ids,
            (physical_ids[0], "query-fresh"),
        )
    validate_lpo_rc_physical_id_disjointness(
        physical_ids,
        ("query-fresh-a", "query-fresh-b"),
    )


def test_builder_has_no_ground_role_or_query_surface() -> None:
    signature = inspect.signature(build_lpo_rc_qknn_state)
    assert tuple(signature.parameters) == (
        "bank",
        "support_zid",
        "support_labels",
        "registered_classes",
        "metric",
        "support_physical_ids",
        "split_handle",
    )
    assert not any(
        token in parameter.lower()
        for parameter in signature.parameters
        for token in ("ground", "role", "truth", "quota", "global", "source")
    )
    support, labels, physical_ids, bank, metric, handle, _ = _build(1)
    with pytest.raises(TypeError):
        build_lpo_rc_qknn_state(
            bank,
            support,
            labels,
            CLASSES,
            metric=metric,
            support_physical_ids=physical_ids,
            split_handle=handle,
            ground_bundle=object(),
        )
    with pytest.raises(TypeError):
        build_lpo_rc_qknn_state(
            bank,
            support,
            labels,
            CLASSES,
            metric=metric,
            support_physical_ids=physical_ids,
            split_handle=handle,
            old_new_roles=("old",) * len(CLASSES),
        )


def test_receipt_closes_int8_state_and_exact_resource_counters() -> None:
    _, _, _, bank, metric, handle, state = _build(5)
    receipt = audit_lpo_rc_resource(state, bank, metric, handle)
    support_rows = len(CLASSES) * 5
    assert receipt["schema"] == LPO_RC_RECEIPT_SCHEMA
    assert receipt["head_schema"] == LPO_RC_SCHEMA
    assert receipt["int8_support_vectors_retained"] is True
    assert receipt["raw_iq_retained"] is False
    assert receipt["fp32_support_vector_retained"] is False
    assert receipt["head_deployment_state_bytes"] == 4 * len(CLASSES)
    assert receipt["base_kernel_evaluations_per_query"] == support_rows
    assert receipt["loo_fit_metric_matmul_mac"] == support_rows * support_rows * Z_DIM
    assert receipt["loo_fit_identity_scale_matmul_mac"] == (
        len(CLASSES) * 5 * 5 * Z_DIM
    )
    assert receipt["loo_fit_total_matmul_mac"] == (
        receipt["loo_fit_metric_matmul_mac"]
        + receipt["loo_fit_identity_scale_matmul_mac"]
    )
    assert receipt["query_rows_used_for_fit"] == 0
    assert receipt["query_state_updates"] == 0
    assert receipt["query_update_count"] == 0
    workspace = receipt["fit_numeric_workspace"]
    recomputed_phase_bytes = {
        phase: sum(components.values())
        for phase, components in workspace["phase_components"].items()
    }
    assert recomputed_phase_bytes == workspace["phase_bytes"]
    assert workspace["formal_workspace_upper_bound_bytes"] == max(
        recomputed_phase_bytes.values()
    )
    assert workspace["peak_bytes"] == max(
        workspace["phase_bytes"].values()
    )
    # For the same C=4,K=5,d=160 shape, normalize_zid_rows alone has a
    # conservative named-allocation lower bound of 105,800 bytes.  The formal
    # bound must cover it without relying on tracemalloc or a safety multiplier.
    assert workspace["peak_bytes"] >= 105_800
    assert workspace["passes_formal_workspace_gate"] is True
    assert (
        workspace["formal_workspace_upper_bound_bytes"]
        <= workspace["formal_workspace_budget_bytes"]
    )
    assert "not_measured_allocator_or_process_peak" in workspace["semantics"]
    wire = serialize_lpo_rc_qknn_state(
        state,
        bank=bank,
        metric=metric,
        split_handle=handle,
    )
    assert receipt["head_state_wire_serialized_bytes"] == len(wire)
    assert receipt["head_state_wire_sha256"] == hashlib.sha256(wire).hexdigest()
    scalar = receipt["base_kernel_scalar_logic_counts_per_query"]
    assert scalar["volume_log_ops"] == len(CLASSES)
    assert scalar["radial_log1p_ops"] == support_rows
    assert scalar["logsumexp_exp_ops"] == support_rows
    assert scalar["logsumexp_sum_add_ops"] == len(CLASSES) * 4


def test_state_rejects_bank_binding_drift() -> None:
    _, _, _, bank, metric, handle, state = _build(5)
    other_support, other_labels, _ = _support(5, seed=105733)
    other_bank = build_typed_zid_support_bank(
        other_support,
        other_labels,
        CLASSES,
        config=_lock(5),
    )
    with pytest.raises(LPORCQKNNError, match="binding"):
        score_lpo_rc_qknn_logits(
            state,
            _queries(),
            bank=other_bank,
            metric=metric,
            split_handle=handle,
        )


def test_validated_once_split_handle_is_mandatory_and_digest_bound() -> None:
    support, labels, physical_ids, bank, metric, handle, state = _build(5)
    with pytest.raises(TypeError):
        build_lpo_rc_qknn_state(
            bank,
            support,
            labels,
            CLASSES,
            metric=metric,
            support_physical_ids=physical_ids,
        )
    with pytest.raises(TypeError):
        score_lpo_rc_qknn_logits(
            state,
            _queries(),
            bank=bank,
            metric=metric,
        )
    wrong_root = replace(handle, support_physical_root_sha256="8" * 64)
    with pytest.raises(LPORCProtocolHandleError, match="support physical root"):
        build_lpo_rc_qknn_state(
            bank,
            support,
            labels,
            CLASSES,
            metric=metric,
            support_physical_ids=physical_ids,
            split_handle=wrong_root,
        )
    drifted = replace(handle, capsule_id="9" * 64)
    with pytest.raises(LPORCProtocolHandleError, match="digest drift"):
        score_lpo_rc_qknn_logits(
            state,
            _queries(),
            bank=bank,
            metric=metric,
            split_handle=drifted,
        )
    with pytest.raises(LPORCProtocolHandleError, match="disjoint"):
        replace(handle, support_query_disjoint=False)


def test_receipt_is_recursively_immutable_audit_is_deep_copy_and_tamper_fails() -> None:
    _, _, _, bank, metric, handle, state = _build(5)
    with pytest.raises(TypeError):
        state.receipt["fit_numeric_workspace"]["peak_bytes"] = 0
    first = audit_lpo_rc_resource(state, bank, metric, handle)
    first["fit_numeric_workspace"]["loo_components"]["decoded_support_fp64"] = 0
    second = audit_lpo_rc_resource(state, bank, metric, handle)
    assert second["fit_numeric_workspace"]["loo_components"]["decoded_support_fp64"] > 0

    tampered = audit_lpo_rc_resource(state, bank, metric, handle)
    tampered["query_update_count"] = 1
    object.__setattr__(state, "receipt", MappingProxyType(tampered))
    with pytest.raises(LPORCQKNNError, match="receipt hash drift"):
        score_lpo_rc_qknn_logits(
            state,
            _queries(),
            bank=bank,
            metric=metric,
            split_handle=handle,
        )


def test_rank1_metric_uses_exact_repeated_precision_cosine_mac_count() -> None:
    support, labels, physical_ids = _support(5)
    config = _lock(5)
    bank = build_typed_zid_support_bank(
        support,
        labels,
        CLASSES,
        config=config,
    )
    metric = _rank1_metric(config)
    handle = _split_handle(physical_ids)
    state = build_lpo_rc_qknn_state(
        bank,
        support,
        labels,
        CLASSES,
        metric=metric,
        support_physical_ids=physical_ids,
        split_handle=handle,
    )
    receipt = audit_lpo_rc_resource(state, bank, metric, handle)
    rows = len(CLASSES) * 5
    rank = 1
    score_call_mac = (
        (rows - 1) * Z_DIM
        + rank * Z_DIM
        + (rows - 1) * rank * Z_DIM
        + (rows - 1) * rank
    )
    reconstruction_call_mac = Z_DIM + rank * Z_DIM + rank * Z_DIM + rank
    assert receipt["loo_fit_metric_matmul_mac"] == rows * (
        score_call_mac + reconstruction_call_mac
    )
    logits = score_lpo_rc_qknn_logits(
        state,
        _queries(),
        bank=bank,
        metric=metric,
        split_handle=handle,
    )
    assert np.isfinite(logits).all()


def test_physical_loo_constructively_removes_one_support_per_fit_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cvsrffi.stage2_lpo_rc_qknn as module

    observed: list[tuple[int, ...]] = []
    original_score = module._score_with_support

    def _recording_score(**kwargs):
        observed.append(tuple(kwargs["support_counts"]))
        return original_score(**kwargs)

    monkeypatch.setattr(module, "_score_with_support", _recording_score)
    _build(5)
    assert len(observed) == len(CLASSES) * 5
    assert all(sum(counts) == len(CLASSES) * 5 - 1 for counts in observed)
    assert all(sorted(counts) == [4, 5, 5, 5] for counts in observed)


def test_nonfinite_support_and_state_tamper_fail_closed() -> None:
    support, labels, physical_ids, bank, metric, handle, state = _build(5)
    invalid = support.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_lpo_rc_qknn_state(
            bank,
            invalid,
            labels,
            CLASSES,
            metric=metric,
            support_physical_ids=physical_ids,
            split_handle=handle,
        )
    object.__setattr__(
        state,
        "bias_fp16",
        np.full(len(CLASSES), np.float16(np.nan), dtype=np.float16),
    )
    with pytest.raises(LPORCQKNNError, match="tamper"):
        score_lpo_rc_qknn_logits(
            state,
            _queries(),
            bank=bank,
            metric=metric,
            split_handle=handle,
        )
