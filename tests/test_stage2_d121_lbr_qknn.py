from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d121_lbr_qknn as d121
from cvsrffi import stage2_zid_student_t_qknn as qknn
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
)


CLASSES = ("alpha", "bravo", "charlie")


def _lock(k: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=3.0,
        kernel_effective_dim=160,
        kernel_volume_gamma=1.0,
        shared_h0=0.2,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="a" * 64,
        quantization_margin_audit_sha256="b" * 64,
    )


def _bank_order_ids(
    support: np.ndarray,
    labels: tuple[str, ...],
    physical_ids: tuple[str, ...],
    registry: tuple[str, ...],
    lock: Phase1ZIDStudentTLock,
):
    bank = build_typed_zid_support_bank(support, labels, registry, config=lock)
    normalized = qknn.normalize_zid_rows(support)
    codes, scales, _decoded = qknn._quantize_rows(normalized)
    class_map = {label: index for index, label in enumerate(registry)}
    class_indices = np.asarray([class_map[label] for label in labels], dtype=np.int16)
    order = qknn._canonical_order(codes, scales, class_indices)
    return bank, tuple(physical_ids[int(index)] for index in order)


def _random_bank(k: int, registry: tuple[str, ...] = CLASSES):
    rng = np.random.default_rng(12100 + k)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for class_index, label in enumerate(registry):
        centre = np.zeros(160, dtype=np.float32)
        centre[10 + class_index] = np.float32(1.0)
        for shot in range(k):
            value = centre + rng.normal(0.0, 0.015, 160).astype(np.float32)
            rows.append(value)
            labels.append(label)
            physical_ids.append(f"p-{label}-{shot:02d}")
    support = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    lock = _lock(k)
    bank, bank_order_ids = _bank_order_ids(
        support, tuple(labels), tuple(physical_ids), registry, lock
    )
    metric = identity_shared_psd_metric(config=lock)
    query = np.ascontiguousarray(
        np.stack(
            [
                support[0] + np.float32(0.003),
                support[k] + np.float32(0.004),
                support[2 * k] + np.float32(0.005),
            ]
        ),
        dtype=np.float32,
    )
    return bank, bank_order_ids, metric, query


def test_lbr_single_hop_formula_and_query_zero_state() -> None:
    bank, physical_ids, metric, query = _random_bank(5)
    state = d121.build_lbr_qknn_state(bank, physical_ids, metric=metric)
    trace = d121.score_lbr_qknn_trace(state, bank, query, metric=metric)

    expected = trace.base_support_logits_fp64 - np.logaddexp(
        0.0,
        trace.base_support_logits_fp64[:, state.rival_indices_uint16]
        - trace.base_support_logits_fp64,
    )
    np.testing.assert_allclose(trace.lbr_support_logits_fp64, expected, rtol=0.0, atol=0.0)
    assert trace.class_logits_fp32.shape == (len(query), len(CLASSES))
    assert np.isfinite(trace.class_logits_fp32).all()
    assert d121.unique_lbr_argmax(trace.class_logits_fp32, CLASSES)

    audit = d121.audit_lbr_qknn_state(state)
    assert audit["query_rows_used_for_fit"] == 0
    assert audit["query_state_updates"] == 0
    assert audit["query_selection_count"] == 0
    assert state.resource_receipt["parameter_scan_count"] == 0
    assert state.resource_receipt["persistent_numeric_bytes"] == 2 * bank.support_row_count


def test_k1_k5_k10_all_build_foreign_rivals_and_score() -> None:
    for k in (1, 5, 10):
        bank, physical_ids, metric, query = _random_bank(k)
        state = d121.build_lbr_qknn_state(bank, physical_ids, metric=metric)
        support_classes = np.asarray(bank.class_indices_int16, dtype=np.int16)
        rival_classes = support_classes[state.rival_indices_uint16.astype(np.int64)]
        assert np.all(rival_classes != support_classes)
        assert state.rival_indices_uint16.dtype == np.uint16
        assert not state.rival_indices_uint16.flags.writeable
        result = d121.score_lbr_qknn_logits(state, bank, query, metric=metric)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()


def test_label_permutation_equivariance() -> None:
    bank_a, ids_a, metric_a, query = _random_bank(5, CLASSES)
    state_a = d121.build_lbr_qknn_state(bank_a, ids_a, metric=metric_a)
    scores_a = d121.score_lbr_qknn_logits(state_a, bank_a, query, metric=metric_a)

    permuted = tuple(reversed(CLASSES))
    rng = np.random.default_rng(12105)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for class_index, label in enumerate(CLASSES):
        centre = np.zeros(160, dtype=np.float32)
        centre[10 + class_index] = np.float32(1.0)
        for shot in range(5):
            value = centre + rng.normal(0.0, 0.015, 160).astype(np.float32)
            rows.append(value)
            labels.append(label)
            physical_ids.append(f"p-{label}-{shot:02d}")
    support = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    lock_b = _lock(5)
    bank_b, ids_b = _bank_order_ids(
        support, tuple(labels), tuple(physical_ids), permuted, lock_b
    )
    metric_b = identity_shared_psd_metric(config=lock_b)
    state_b = d121.build_lbr_qknn_state(bank_b, ids_b, metric=metric_b)
    scores_b = d121.score_lbr_qknn_logits(state_b, bank_b, query, metric=metric_b)

    # The helper uses the same seeded support construction as _random_bank(5).
    for label in CLASSES:
        np.testing.assert_allclose(
            scores_a[:, CLASSES.index(label)],
            scores_b[:, permuted.index(label)],
            rtol=0.0,
            atol=0.0,
        )
    assert d121.unique_lbr_argmax(scores_a, CLASSES) == d121.unique_lbr_argmax(
        scores_b, permuted
    )


def test_tied_foreign_distance_uses_content_hash_then_fails_closed(monkeypatch) -> None:
    support = np.zeros((3, 160), dtype=np.float32)
    support[0, 0] = 1.0
    support[1, 1] = 1.0
    support[2, 2] = 1.0
    labels = ("alpha", "bravo", "charlie")
    physical = ("a-id", "b-id", "c-id")
    lock = _lock(1)
    bank, ids = _bank_order_ids(support, labels, physical, CLASSES, lock)
    metric = identity_shared_psd_metric(config=lock)
    position = {value: index for index, value in enumerate(ids)}
    monkeypatch.setattr(d121, "_sha256_text", lambda _value: "0" * 64)
    content_by_id = {"a-id": "a" * 64, "b-id": "b" * 64, "c-id": "c" * 64}
    content = tuple(content_by_id[value] for value in ids)
    state = d121.build_lbr_qknn_state(
        bank, ids, metric=metric, support_content_hashes_in_bank_order=content
    )
    assert ids[int(state.rival_indices_uint16[position["a-id"]])] == "b-id"

    with pytest.raises(d121.RivalTieUnresolvedError, match="RIVAL_TIE_UNRESOLVED"):
        d121.build_lbr_qknn_state(
            bank,
            ids,
            metric=metric,
            support_content_hashes_in_bank_order=("d" * 64,) * 3,
        )


def test_final_class_tie_and_input_surface_fail_closed() -> None:
    with pytest.raises(
        d121.ClassScoreTieUnresolvedError, match="CLASS_SCORE_TIE_UNRESOLVED"
    ):
        d121.unique_lbr_argmax(
            np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32), CLASSES
        )
    build_parameters = inspect.signature(d121.build_lbr_qknn_state).parameters
    score_parameters = inspect.signature(d121.score_lbr_qknn_trace).parameters
    assert "query" not in build_parameters
    assert not {"truth", "role", "quota", "selection"} & set(build_parameters)
    assert not {"truth", "role", "quota", "selection"} & set(score_parameters)
