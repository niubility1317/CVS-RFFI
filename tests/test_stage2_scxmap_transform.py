from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.stage2_scxmap_transform import (
    CONTEXT_DIM,
    SCXMapError,
    Z_DIM,
    audit_scxmap_resources,
    build_phase1_scxmap_lock,
    fit_scxmap_state,
    transform_scxmap_rows,
)


SHA = "3" * 64


def _unit(index: int) -> np.ndarray:
    row = np.zeros(Z_DIM, dtype=np.float32)
    row[index] = 1.0
    return row


def _lock():
    projection = np.zeros((CONTEXT_DIM, Z_DIM), dtype=np.float32)
    basis = np.zeros((CONTEXT_DIM, Z_DIM), dtype=np.float32)
    for index in range(CONTEXT_DIM):
        projection[index, index] = 1.0
        basis[index, index] = 1.0
    lock, audit = build_phase1_scxmap_lock(
        ground_classes=("g0", "g1", "g2"),
        zdom_center=np.zeros(Z_DIM, dtype=np.float32),
        zdom_scale=np.ones(Z_DIM, dtype=np.float32),
        receiver_projection=projection,
        context_to_shift=np.eye(CONTEXT_DIM, dtype=np.float32),
        zid_basis=basis,
        ground_anchors=np.stack([_unit(8), _unit(9), _unit(10)]),
        ridge_per_row=0.0,
        shrink_tau=0.0,
        beta_max=2.0,
        source_receipt_sha256=SHA,
    )
    return lock, audit


def _support():
    labels = ("g0", "g1", "g2")
    zdom = np.zeros((3, Z_DIM), dtype=np.float32)
    zdom[:, 0] = np.asarray((0.25, 0.5, 0.75), dtype=np.float32)
    zid = np.stack([_unit(8), _unit(9), _unit(10)])
    zid[:, 0] += zdom[:, 0]
    return zid.astype(np.float32), zdom, labels


def test_k1_ground_anchor_fit_is_positive_and_corrects_toward_ground():
    lock, audit = _lock()
    zid, zdom, labels = _support()
    state = fit_scxmap_state(
        lock, zid, zdom, labels, support_receipt_sha256="4" * 64
    )
    transformed = transform_scxmap_rows(lock, state, zid, zdom)
    before = np.sum(
        zid / np.linalg.norm(zid, axis=1, keepdims=True)
        * np.stack([_unit(8), _unit(9), _unit(10)]),
        axis=1,
    )
    after = np.sum(
        transformed * np.stack([_unit(8), _unit(9), _unit(10)]), axis=1
    )
    assert state.old_support_rows == 3
    assert state.old_class_count == 3
    assert 0.0 < state.beta_fp32 <= lock.beta_max
    assert np.all(after > before)
    assert audit["persistent_array_bytes"] < 256 * 1024


def test_support_permutation_and_query_batch_order_are_invariant():
    lock, _ = _lock()
    zid, zdom, labels = _support()
    state = fit_scxmap_state(
        lock, zid, zdom, labels, support_receipt_sha256="4" * 64
    )
    order = np.asarray((2, 0, 1))
    permuted = fit_scxmap_state(
        lock,
        zid[order],
        zdom[order],
        [labels[index] for index in order],
        support_receipt_sha256="4" * 64,
    )
    assert permuted.beta_fp32 == state.beta_fp32
    query = transform_scxmap_rows(lock, state, zid, zdom)
    query_permuted = transform_scxmap_rows(lock, state, zid[order], zdom[order])
    assert np.array_equal(query[order], query_permuted)
    singles = np.concatenate(
        [
            transform_scxmap_rows(lock, state, zid[index : index + 1], zdom[index : index + 1])
            for index in range(len(zid))
        ],
        axis=0,
    )
    assert np.array_equal(query, singles)


def test_identity_config_and_zero_domain_correction_are_exact():
    lock, _ = _lock()
    zid, zdom, labels = _support()
    state = fit_scxmap_state(
        lock, zid, zdom, labels, support_receipt_sha256="4" * 64
    )
    disabled = transform_scxmap_rows(lock, state, zid, zdom, enabled=False)
    assert disabled.dtype == np.float32
    assert np.array_equal(disabled, zid)
    zero_domain = np.zeros_like(zdom)
    corrected = transform_scxmap_rows(lock, state, zid, zero_domain)
    assert np.array_equal(corrected, zid)
    negative = np.stack([_unit(8), _unit(9), _unit(10)])
    negative[:, 0] -= zdom[:, 0]
    zero_state = fit_scxmap_state(
        lock,
        negative.astype(np.float32),
        zdom,
        labels,
        support_receipt_sha256="5" * 64,
    )
    assert zero_state.beta_fp32 == 0.0
    assert np.array_equal(
        transform_scxmap_rows(lock, zero_state, negative.astype(np.float32), zdom),
        negative.astype(np.float32),
    )


def test_new_support_is_rejected_from_da_fit_and_lock_is_immutable():
    lock, _ = _lock()
    zid, zdom, labels = _support()
    with pytest.raises(SCXMapError, match="target-old support only"):
        fit_scxmap_state(
            lock,
            np.concatenate([zid, zid[:1]]),
            np.concatenate([zdom, zdom[:1]]),
            labels + ("new",),
            support_receipt_sha256="4" * 64,
        )
    with pytest.raises(ValueError):
        lock.ground_anchor_qint8[0, 0] = 0


def test_resource_ledger_and_typed_negative_inputs():
    lock, _ = _lock()
    zid, zdom, labels = _support()
    state = fit_scxmap_state(
        lock, zid, zdom, labels, support_receipt_sha256="4" * 64
    )
    resource = audit_scxmap_resources(lock, state)
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["optimizer_steps"] == 0
    assert resource["effective_rank"] == CONTEXT_DIM
    assert resource["query_matmul_mac_per_row"] == 1296
    assert resource["persistent_state_wire_bytes"] < 256 * 1024
    assert resource["query_elementwise_ops_per_row"] > 0
    with pytest.raises(SCXMapError, match="finite float32"):
        transform_scxmap_rows(lock, state, zid.astype(np.float64), zdom)
    with pytest.raises(SCXMapError, match="every ground-old class"):
        fit_scxmap_state(
            lock,
            zid[:1],
            zdom[:1],
            labels[:1],
            support_receipt_sha256="4" * 64,
        )
    with pytest.raises(SCXMapError, match="same legal K"):
        fit_scxmap_state(
            lock,
            np.concatenate([zid, zid[:1]]),
            np.concatenate([zdom, zdom[:1]]),
            labels + ("g0",),
            support_receipt_sha256="4" * 64,
        )
