from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_bundle import build_rxid_metabias4_bundle
from cvsrffi.stage2_d105_cbrc import (
    compute_d105_bundle_receipt_root,
    compute_d105_bundle_validator_receipt,
    compute_d105_support_binding_root,
    make_d105_cbrc_bundle_handle,
)
from cvsrffi.stage2_d105_four_arm import (
    ARMS,
    D105FourArmError,
    audit_d105_four_arm_resources,
    build_d105_four_arm_state,
    score_d105_four_arm_logits,
)
from cvsrffi.stage2_lpo_rc_qknn import TypedValidatedOnceP2SplitHandle
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    _canonical_sha256,
)


OLD = ("old-a", "old-b")
NEW = ("new-a", "new-b")
CLASSES = OLD + NEW
HASHES = tuple(f"{index:x}" * 64 for index in range(1, 10))


def _bundle():
    rng = np.random.default_rng(105731)
    u = np.zeros((32, 160), dtype=np.float32)
    u[:, :32] = np.eye(32, dtype=np.float32)
    b = rng.normal(0.0, 0.03, (160, 4)).astype(np.float32)
    g = np.zeros((6, 32), dtype=np.float32)
    for index in range(6):
        g[index, index] = 1.0
        g[index, (index + 7) % 32] = 0.25
    t = np.asarray(
        [
            [0.18, 0.06, -0.05, 0.09],
            [0.15, 0.04, -0.03, 0.08],
            [0.12, 0.05, -0.02, 0.06],
            [0.16, 0.07, -0.04, 0.10],
            [0.10, 0.03, -0.01, 0.05],
            [0.14, 0.06, -0.02, 0.07],
        ],
        dtype=np.float32,
    )
    return build_rxid_metabias4_bundle(
        u,
        b,
        g,
        t,
        np.full((6, 4), 4.0, dtype=np.float32),
        np.full(6, 1.7, dtype=np.float32),
        cell_min_physical_count=np.full(6, 2, dtype=np.int16),
        cell_class_count=np.full(6, 3, dtype=np.int16),
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


def _bundle_handle(bundle):
    receipt_root = compute_d105_bundle_receipt_root(bundle)
    validator_receipt = compute_d105_bundle_validator_receipt(
        validated_bundle_id_sha256=HASHES[8],
        expected_content_root_sha256=bundle.content_root_sha256,
        checkpoint_sha256=bundle.checkpoint_sha256,
        runtime_sha256=bundle.runtime_sha256,
        method_lock_sha256=bundle.method_lock_sha256,
        receipt_root_sha256=receipt_root,
    )
    return make_d105_cbrc_bundle_handle(
        bundle,
        validated_bundle_id_sha256=HASHES[8],
        validator_receipt_sha256=validator_receipt,
        expected_content_root_sha256=bundle.content_root_sha256,
    )


def _lock(k: int) -> Phase1ZIDStudentTLock:
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
        phase1_lodo_receipt_sha256="a" * 64,
        quantization_margin_audit_sha256="b" * 64,
    )


def _support(k: int):
    rng = np.random.default_rng(105800 + k)
    pre_relu: list[np.ndarray] = []
    zdom: list[np.ndarray] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for class_index, label in enumerate(CLASSES):
        for shot in range(k):
            pre = rng.normal(0.08, 0.025, 160).astype(np.float32)
            pre[20 + class_index] += np.float32(0.9)
            domain = rng.normal(0.0, 0.02, 160).astype(np.float32)
            domain[class_index] += np.float32(1.0)
            pre_relu.append(pre)
            zdom.append(domain)
            labels.append(label)
            physical_ids.append(f"support-{label}-{shot}")
    return (
        np.asarray(pre_relu, dtype=np.float32),
        np.asarray(zdom, dtype=np.float32),
        tuple(labels),
        tuple(physical_ids),
    )


def _query():
    values = np.random.default_rng(105900).normal(
        0.08, 0.08, (7, 160)
    ).astype(np.float32)
    values[:, 20:24] += np.float32(0.3)
    identifiers = tuple(f"query-{index}" for index in range(len(values)))
    return values, identifiers


def _build(k: int):
    bundle = _bundle()
    pre_relu, zdom, labels, support_ids = _support(k)
    query, query_ids = _query()
    split_handle = TypedValidatedOnceP2SplitHandle(
        capsule_id="c" * 64,
        split_id="d" * 64,
        validator_receipt_sha256="e" * 64,
        support_physical_root_sha256=_canonical_sha256(sorted(support_ids)),
        query_physical_root_sha256=_canonical_sha256(sorted(query_ids)),
        support_query_disjoint=True,
    )
    support_receipt = compute_d105_support_binding_root(
        pre_relu,
        zdom,
        labels,
        support_ids,
        CLASSES,
        OLD,
        NEW,
        active_k=k,
        stage="S_C",
    )
    state = build_d105_four_arm_state(
        bundle,
        _bundle_handle(bundle),
        pre_relu,
        zdom,
        labels,
        support_ids,
        CLASSES,
        OLD,
        NEW,
        config=_lock(k),
        split_handle=split_handle,
        active_k=k,
        stage="S_C",
        support_receipt_sha256=support_receipt,
    )
    return state, query, query_ids


@pytest.mark.parametrize("k", [1, 5])
def test_four_arm_scoring_is_complete_and_read_only(k: int) -> None:
    state, query, query_ids = _build(k)
    before = state.receipt_sha256
    result = score_d105_four_arm_logits(
        state, query, query_physical_ids=query_ids
    )
    assert tuple(result.by_arm) == ARMS
    assert all(value.shape == (len(query), len(CLASSES)) for value in result.by_arm.values())
    assert result.state_receipt_sha256 == before == state.receipt_sha256
    assert all(not value.flags.writeable for value in result.by_arm.values())


def test_k1_head_arms_are_exact_identities() -> None:
    state, query, query_ids = _build(1)
    result = score_d105_four_arm_logits(
        state, query, query_physical_ids=query_ids
    )
    np.testing.assert_array_equal(result.m_head, result.m0)
    np.testing.assert_array_equal(result.m_joint, result.m_da)
    assert result.m_head.tobytes() == result.m0.tobytes()
    assert result.m_joint.tobytes() == result.m_da.tobytes()


def test_query_order_is_independent_and_bound_to_the_same_root() -> None:
    state, query, query_ids = _build(5)
    first = score_d105_four_arm_logits(
        state, query, query_physical_ids=query_ids
    )
    chunked = score_d105_four_arm_logits(
        state,
        query,
        query_physical_ids=query_ids,
        chunk_size=3,
    )
    second = score_d105_four_arm_logits(
        state,
        query[::-1],
        query_physical_ids=query_ids[::-1],
        chunk_size=2,
    )
    for arm in ARMS:
        np.testing.assert_array_equal(first.by_arm[arm], chunked.by_arm[arm])
        np.testing.assert_array_equal(
            first.by_arm[arm],
            second.by_arm[arm][::-1],
        )


def test_query_root_and_state_tamper_fail_closed() -> None:
    state, query, query_ids = _build(5)
    with pytest.raises(D105FourArmError, match="query physical root"):
        score_d105_four_arm_logits(
            state,
            query,
            query_physical_ids=query_ids[:-1] + ("query-replacement",),
        )
    with pytest.raises(D105FourArmError, match="state receipt drift"):
        score_d105_four_arm_logits(
            replace(state, receipt_sha256="f" * 64),
            query,
            query_physical_ids=query_ids,
        )
    with pytest.raises(D105FourArmError, match="chunk_size"):
        score_d105_four_arm_logits(
            state,
            query,
            query_physical_ids=query_ids,
            chunk_size=0,
        )


def test_four_arm_resource_receipt_keeps_component_units_and_zero_query_update() -> None:
    state, _, _ = _build(5)
    receipt = audit_d105_four_arm_resources(state)
    assert receipt["query_state_updates"] == 0
    assert receipt["query_rows_used_for_fit"] == 0
    assert receipt["da"]["query_state_updates"] == 0
    assert receipt["base_head"]["query_update_count"] == 0
    assert receipt["da_head"]["query_update_count"] == 0
