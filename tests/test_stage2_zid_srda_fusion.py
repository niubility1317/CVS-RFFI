from __future__ import annotations

import ast
import copy
from dataclasses import replace
import inspect
import json
import struct

import numpy as np
import pytest

import cvsrffi.stage2_zid_srda_fusion as srda
import cvsrffi.stage2_zid_student_t_qknn as zid


CLASSES = ("cls_a", "cls_b", "cls_c")


def _a_lock(k_shot: int) -> zid.Phase1ZIDStudentTLock:
    return zid.Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.5,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _b_lock(
    k_shot: int,
    *,
    alpha: float = 0.35,
    target_rank: int = 2,
    ground_weight: float = 0.5,
    ground_rank: int = 3,
    a_lock: zid.Phase1ZIDStudentTLock | None = None,
) -> srda.Phase1ZIDSRDALock:
    a_config = _a_lock(k_shot) if a_lock is None else a_lock
    identity = zid.identity_shared_psd_metric(config=a_config)
    prior = _ground(ground_rank) if ground_weight != 0.0 else None
    return srda.Phase1ZIDSRDALock(
        active_k=k_shot,
        sigma0_sq=0.7,
        nu0=8.0,
        target_rank=target_rank,
        lambda_relative=0.05,
        rda_temperature=1.1,
        ground_weight=ground_weight,
        ground_prior_rank=0 if prior is None else prior.rank,
        alpha_phase1=alpha,
        a_component_sha256="a" * 64,
        a_config_lock_digest=a_config.lock_digest,
        a_identity_metric_receipt_sha256=identity.metric_receipt_sha256,
        ground_prior_receipt_sha256=(
            srda.ZERO_SHA256 if prior is None else prior.prior_receipt_sha256
        ),
        ground_source_receipt_sha256=(
            srda.ZERO_SHA256
            if prior is None
            else prior.provenance.source_receipt_sha256
        ),
        phase1_lodo_receipt_sha256="b" * 64,
        quantization_margin_audit_sha256="c" * 64,
    )


def _support(k_shot: int, *, seed: int = 731) -> tuple[np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for class_index, label in enumerate(CLASSES):
        center = np.zeros(zid.Z_DIM, dtype=np.float32)
        center[class_index] = 1.0
        center[10 + class_index] = 0.35
        for shot in range(k_shot):
            noise = rng.normal(scale=0.025, size=zid.Z_DIM).astype(np.float32)
            noise[40 + shot % 7] += np.float32(0.02 * (shot + 1))
            rows.append(center + noise)
            labels.append(label)
    return np.asarray(rows, dtype=np.float32), tuple(labels)


def _bank(k_shot: int, *, seed: int = 731) -> zid.TypedINT8ZIDSupportBank:
    support, labels = _support(k_shot, seed=seed)
    return zid.build_typed_zid_support_bank(
        support, labels, CLASSES, config=_a_lock(k_shot)
    )


def _ground(rank: int = 3) -> srda.TypedZIDSharedCovariancePrior:
    basis = np.eye(zid.Z_DIM, rank, dtype=np.float32)
    spectrum = np.linspace(1.0, 2.0, rank, dtype=np.float32)
    return srda.build_typed_shared_covariance_prior(
        basis,
        spectrum,
        provenance=srda.TypedZIDGroundPriorProvenance(
            source_receipt_sha256="d" * 64
        ),
    )


def test_k1_has_exact_zero_target_dof_and_rank_even_with_ground() -> None:
    bank = _bank(1)
    state = srda.build_zid_srda_state(
        bank, _ground(3), lock=_b_lock(1, target_rank=0)
    )

    assert state.fit_audit["n_residual_degrees"] == 0
    assert state.fit_audit["target_scatter_exact_zero"] is True
    assert state.fit_audit["target_rank_actual"] == 0
    assert state.fit_audit["ground_rank_actual"] == 3
    assert state.fit_audit["woodbury_rank_total"] == 3
    assert state.fit_audit["shrinkage"] == 0.0
    assert state.fit_audit["query_rows_used_for_fit"] == 0


def test_class_balanced_scatter_matches_hand_fixture_and_mean_is_not_normalized() -> None:
    support = np.zeros((4, zid.Z_DIM), dtype=np.float64)
    support[0, :2] = (1.0, 0.0)
    support[1, :2] = (0.0, 1.0)
    support[2, :2] = (1.0, 0.0)
    support[3, :2] = (0.0, -1.0)
    indices = np.asarray([0, 0, 1, 1], dtype=np.int16)

    means, residual, covariance, nres = srda._class_balanced_scatter(
        support, indices, 2, 2
    )

    expected = np.zeros((zid.Z_DIM, zid.Z_DIM), dtype=np.float64)
    expected[0, 0] = 0.5
    expected[1, 1] = 0.5
    assert nres == 2
    assert np.allclose(covariance, expected, atol=0.0, rtol=0.0)
    assert np.allclose(means[0, :2], (0.5, 0.5), atol=0.0, rtol=0.0)
    assert not np.isclose(np.linalg.norm(means[0]), 1.0)
    assert np.allclose(np.sum(residual, axis=0), 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("rank", [0, 1, 2, 4, 6])
def test_woodbury_matches_dense_inverse(rank: int) -> None:
    rng = np.random.default_rng(91 + rank)
    rows = rng.normal(size=(5, zid.Z_DIM))
    factor = rng.normal(scale=0.03, size=(zid.Z_DIM, rank))
    diagonal = 0.7
    actual = srda.woodbury_precision_apply(rows, diagonal, factor)
    dense = diagonal * np.eye(zid.Z_DIM) + factor @ factor.T
    expected = np.linalg.solve(dense, rows.T).T
    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_ground_prior_rank_orthogonality_and_typed_bounds() -> None:
    prior = _ground(4)
    decoded = srda.decode_shared_covariance_prior(prior)
    assert decoded.shape == (zid.Z_DIM, 4)
    assert np.allclose(decoded.T @ decoded, np.eye(4), atol=2e-2, rtol=0.0)

    with pytest.raises(srda.ZIDSRDAError):
        _ground(5)
    nonorthogonal = np.eye(zid.Z_DIM, 2, dtype=np.float32)
    nonorthogonal[:, 1] = nonorthogonal[:, 0]
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_typed_shared_covariance_prior(
            nonorthogonal,
            np.ones(2, dtype=np.float32),
            provenance=srda.TypedZIDGroundPriorProvenance(
                source_receipt_sha256="d" * 64
            ),
        )
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_typed_shared_covariance_prior(
            np.eye(zid.Z_DIM, 2, dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            provenance=srda.TypedZIDGroundPriorProvenance(
                source_receipt_sha256="d" * 64
            ),
        )


def test_k5_target_rank_is_positive_capped_and_total_rank_at_most_six() -> None:
    bank = _bank(5)
    state = srda.build_zid_srda_state(
        bank, _ground(4), lock=_b_lock(5, ground_rank=4)
    )
    assert 0 <= state.fit_audit["target_rank_actual"] <= 2
    assert state.fit_audit["ground_rank_actual"] == 4
    assert state.fit_audit["woodbury_rank_total"] <= 6
    assert all(value > 0.0 for value in state.fit_audit["positive_target_eigenvalues"])
    assert state.fit_audit["class_balanced_scatter"] is True
    assert state.fit_audit["ground_class_mean_or_logit_access"] is False


def test_support_row_permutation_produces_exact_same_bank_and_state_wire() -> None:
    support, labels = _support(5)
    order = np.random.default_rng(99).permutation(len(support))
    first_bank = zid.build_typed_zid_support_bank(
        support, labels, CLASSES, config=_a_lock(5)
    )
    second_bank = zid.build_typed_zid_support_bank(
        support[order], tuple(labels[index] for index in order), CLASSES, config=_a_lock(5)
    )
    first = srda.build_zid_srda_state(first_bank, _ground(), lock=_b_lock(5))
    second = srda.build_zid_srda_state(second_bank, _ground(), lock=_b_lock(5))
    assert first_bank.bank_receipt_sha256 == second_bank.bank_receipt_sha256
    assert srda.serialize_zid_srda_state(first) == srda.serialize_zid_srda_state(second)


def test_alpha_zero_is_bit_exact_A_and_skips_rda_branch(monkeypatch) -> None:
    bank = _bank(5)
    state = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, alpha=0.0, ground_weight=0.0)
    )
    query = np.random.default_rng(5).normal(size=(7, zid.Z_DIM)).astype(np.float32)
    identity = zid.identity_shared_psd_metric(config=bank.config)
    expected = zid.softmax_probabilities(
        zid.score_zid_student_t_logits(bank, query, metric=identity), config=bank.config
    )

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("alpha-zero must skip the RDA query branch")

    monkeypatch.setattr(srda, "_score_rda_logits", _must_not_run)
    result = srda.fuse_zid_qknn_srda(bank, state, query)

    assert result.rda_probability_fp32 is None
    assert np.array_equal(result.a_probability_fp32, expected)
    assert np.array_equal(result.fused_probability_fp32, expected)
    assert result.fused_probability_fp32.dtype == expected.dtype
    assert result.fused_probability_fp32.tobytes() == expected.tobytes()
    assert result.audit["rda_branch_evaluated"] is False
    assert result.audit["a_state_sha256_before"] == result.audit["a_state_sha256_after"]
    assert state.resource_audit["incremental_rda_linear_matmul_mac_per_query"] == 0


def test_alpha_one_is_exact_rda_and_mixture_is_phase1_locked() -> None:
    bank = _bank(5)
    query = np.random.default_rng(8).normal(size=(6, zid.Z_DIM)).astype(np.float32)
    one = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, alpha=1.0, ground_weight=0.0)
    )
    mixed = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, alpha=0.25, ground_weight=0.0)
    )
    one_result = srda.fuse_zid_qknn_srda(bank, one, query)
    mixed_result = srda.fuse_zid_qknn_srda(bank, mixed, query)

    assert np.array_equal(one_result.fused_probability_fp32, one_result.rda_probability_fp32)
    expected = (
        0.75 * mixed_result.a_probability_fp32.astype(np.float64)
        + 0.25 * mixed_result.rda_probability_fp32.astype(np.float64)
    ).astype(np.float32)
    assert np.array_equal(mixed_result.fused_probability_fp32, expected)
    assert mixed_result.audit["alpha_source"] == "phase1_lodo_only"
    assert mixed_result.audit["target_support_crossfit"] is False


def test_query_batch_chunk_and_permutation_are_stateless() -> None:
    bank = _bank(5)
    state = srda.build_zid_srda_state(bank, _ground(), lock=_b_lock(5))
    query = np.random.default_rng(108).normal(size=(11, zid.Z_DIM)).astype(np.float32)
    wire_before = srda.serialize_zid_srda_state(state)
    full = srda.fuse_zid_qknn_srda(bank, state, query)
    chunked = np.concatenate(
        [
            srda.fuse_zid_qknn_srda(bank, state, query[index : index + 1]).fused_probability_fp32
            for index in range(len(query))
        ],
        axis=0,
    )
    order = np.random.default_rng(109).permutation(len(query))
    permuted = srda.fuse_zid_qknn_srda(bank, state, query[order])
    inverse = np.argsort(order)
    assert np.array_equal(full.fused_probability_fp32, chunked)
    assert np.array_equal(full.fused_probability_fp32, permuted.fused_probability_fp32[inverse])
    assert srda.serialize_zid_srda_state(state) == wire_before
    assert full.audit["query_state_updates"] == 0
    assert full.audit["query_batch_dependency"] is False


def test_class_permutation_is_equivariant_and_old_new_like_names_do_not_branch() -> None:
    support, labels = _support(5)
    query = np.random.default_rng(311).normal(size=(5, zid.Z_DIM)).astype(np.float32)
    first_bank = zid.build_typed_zid_support_bank(
        support, labels, CLASSES, config=_a_lock(5)
    )
    order = (2, 0, 1)
    permuted_classes = tuple(CLASSES[index] for index in order)
    second_bank = zid.build_typed_zid_support_bank(
        support, labels, permuted_classes, config=_a_lock(5)
    )
    first = srda.build_zid_srda_state(
        first_bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    second = srda.build_zid_srda_state(
        second_bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    first_result = srda.fuse_zid_qknn_srda(first_bank, first, query)
    second_result = srda.fuse_zid_qknn_srda(second_bank, second, query)
    inverse_columns = [permuted_classes.index(label) for label in CLASSES]
    assert np.allclose(
        first_result.fused_probability_fp32,
        second_result.fused_probability_fp32[:, inverse_columns],
        atol=2e-7,
        rtol=0.0,
    )

    renamed = ("old-looking", "new-looking", "unknown-looking")
    renamed_labels = tuple(renamed[CLASSES.index(label)] for label in labels)
    renamed_bank = zid.build_typed_zid_support_bank(
        support, renamed_labels, renamed, config=_a_lock(5)
    )
    renamed_state = srda.build_zid_srda_state(
        renamed_bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    renamed_result = srda.fuse_zid_qknn_srda(renamed_bank, renamed_state, query)
    assert np.allclose(
        first_result.fused_probability_fp32,
        renamed_result.fused_probability_fp32,
        atol=2e-7,
        rtol=0.0,
    )


def test_wire_roundtrip_and_common_tamper_fail_closed() -> None:
    bank = _bank(5)
    state = srda.build_zid_srda_state(bank, _ground(), lock=_b_lock(5))
    wire = srda.serialize_zid_srda_state(state)
    restored = srda.deserialize_zid_srda_state(wire)
    assert srda.serialize_zid_srda_state(restored) == wire

    attacks = [
        b"x" + wire[1:],
        wire[:-1],
        wire + b"x",
        wire[: len(srda.WIRE_MAGIC)] + struct.pack("<I", 4_000_001) + wire[len(srda.WIRE_MAGIC) + 4 :],
    ]
    for attack in attacks:
        with pytest.raises(srda.ZIDSRDAError):
            srda.deserialize_zid_srda_state(attack)

    header_start = len(srda.WIRE_MAGIC) + 4
    header_size = struct.unpack("<I", wire[len(srda.WIRE_MAGIC) : header_start])[0]
    header = json.loads(wire[header_start : header_start + header_size])
    header["alpha_phase1"] = 0.9
    malicious = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered = (
        wire[: len(srda.WIRE_MAGIC)]
        + struct.pack("<I", len(malicious))
        + malicious
        + wire[header_start + header_size :]
    )
    with pytest.raises(srda.ZIDSRDAError):
        srda.deserialize_zid_srda_state(altered)


def test_resource_receipt_is_component_scoped_and_below_numeric_budget() -> None:
    classes = tuple(f"c{index:02d}" for index in range(26))
    rng = np.random.default_rng(602)
    support = rng.normal(size=(26 * 20, zid.Z_DIM)).astype(np.float32)
    labels = tuple(label for label in classes for _ in range(20))
    bank = zid.build_typed_zid_support_bank(
        support, labels, classes, config=_a_lock(20)
    )
    state = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(20, ground_weight=0.0)
    )
    audit = srda.audit_combined_resources(bank, state)

    assert audit["b_incremental_numeric_array_state_bytes"] == 26 * (zid.Z_DIM + 2 + 2)
    assert audit["component_numeric_sum_bytes"] < 256 * 1024
    assert audit["b_incremental_rda_linear_matmul_mac_per_query"] == 26 * zid.Z_DIM
    assert audit["complete_combined_wire_container_available"] is False
    assert audit["not_end_to_end_mac_or_latency"] is True
    assert state.fit_audit["authority_scope"] == srda.DIAGNOSTIC_AUTHORITY_SCOPE
    assert state.quantization_audit["authority_scope"] == srda.DIAGNOSTIC_AUTHORITY_SCOPE
    assert state.resource_audit["authority_scope"] == srda.DIAGNOSTIC_AUTHORITY_SCOPE
    assert "a_wire_bytes_component" not in state.resource_audit


def test_forbidden_dependencies_and_public_signatures_are_absent() -> None:
    source = inspect.getsource(srda)
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for forbidden_module in (
        "stage2_d81",
        "stage2_d99",
        "stage2_d100",
        "stage2_d101",
    ):
        assert all(forbidden_module not in name for name in imports)
    assert "query_zid" not in inspect.signature(srda.build_zid_srda_state).parameters
    assert "support" not in inspect.signature(srda.fuse_zid_qknn_srda).parameters
    assert "alpha_override" not in inspect.signature(srda.fuse_zid_qknn_srda).parameters


def test_active_k_bank_receipt_and_ground_binding_drift_fail_closed() -> None:
    bank = _bank(5)
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_zid_srda_state(
            bank, None, lock=_b_lock(10, ground_weight=0.0)
        )
    state = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    with pytest.raises(ValueError):
        srda.fuse_zid_qknn_srda(
            replace(bank, bank_receipt_sha256="e" * 64),
            state,
            np.ones((1, zid.Z_DIM), dtype=np.float32),
        )
    with pytest.raises(srda.ZIDSRDAError):
        replace(state, ground_prior_receipt_sha256="f" * 64)


def test_k1_rank_and_ground_presence_fail_closed_at_the_lock_boundary() -> None:
    with pytest.raises(srda.ZIDSRDAError):
        _b_lock(1, target_rank=1)
    with pytest.raises(srda.ZIDSRDAError):
        _b_lock(5, alpha=True)

    bank = _bank(5)
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_zid_srda_state(
            bank, None, lock=_b_lock(5, ground_weight=0.2)
        )
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_zid_srda_state(
            bank, _ground(), lock=_b_lock(5, ground_weight=0.0)
        )


@pytest.mark.parametrize("consumer", ["serialize", "fuse", "resource"])
def test_in_memory_state_array_tamper_fails_at_every_public_consumer(
    consumer: str,
) -> None:
    bank = _bank(5)
    state = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    state.weight_codes_qint8.setflags(write=True)
    current = int(state.weight_codes_qint8[0, 0])
    state.weight_codes_qint8[0, 0] = np.int8(current - 1 if current > 0 else current + 1)
    query = np.ones((1, zid.Z_DIM), dtype=np.float32)

    with pytest.raises(srda.ZIDSRDAError):
        if consumer == "serialize":
            srda.serialize_zid_srda_state(state)
        elif consumer == "fuse":
            srda.fuse_zid_qknn_srda(bank, state, query)
        else:
            srda.audit_combined_resources(bank, state)


def test_nested_audit_tamper_and_resigned_semantic_forgery_fail_closed() -> None:
    bank = _bank(5)
    state = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    state.fit_audit["positive_target_eigenvalues"].append(99.0)
    with pytest.raises(srda.ZIDSRDAError):
        srda.serialize_zid_srda_state(state)

    clean = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    forged_fit = dict(clean.fit_audit)
    forged_fit["query_rows_used_for_fit"] = 7
    payload = srda._state_payload(
        classes=clean.classes,
        active_k=clean.active_k,
        codes=clean.weight_codes_qint8,
        scales=clean.weight_scales_fp16,
        bias=clean.bias_fp16,
        alpha=clean.alpha_phase1,
        a_bank_receipt_sha256=clean.a_bank_receipt_sha256,
        a_config_lock_digest=clean.a_config_lock_digest,
        a_metric_receipt_sha256=clean.a_metric_receipt_sha256,
        ground_prior_receipt_sha256=clean.ground_prior_receipt_sha256,
        lock=clean.lock,
        fit_audit=forged_fit,
        quantization_audit=clean.quantization_audit,
        resource_audit=clean.resource_audit,
    )
    with pytest.raises(srda.ZIDSRDAError):
        replace(
            clean,
            fit_audit=forged_fit,
            state_receipt_sha256=srda._canonical_sha256(payload),
        )

    forged_resource = dict(clean.resource_audit)
    forged_resource["incremental_rda_linear_matmul_mac_per_query"] = 123
    payload = srda._state_payload(
        classes=clean.classes,
        active_k=clean.active_k,
        codes=clean.weight_codes_qint8,
        scales=clean.weight_scales_fp16,
        bias=clean.bias_fp16,
        alpha=clean.alpha_phase1,
        a_bank_receipt_sha256=clean.a_bank_receipt_sha256,
        a_config_lock_digest=clean.a_config_lock_digest,
        a_metric_receipt_sha256=clean.a_metric_receipt_sha256,
        ground_prior_receipt_sha256=clean.ground_prior_receipt_sha256,
        lock=clean.lock,
        fit_audit=clean.fit_audit,
        quantization_audit=clean.quantization_audit,
        resource_audit=forged_resource,
    )
    with pytest.raises(srda.ZIDSRDAError):
        replace(
            clean,
            resource_audit=forged_resource,
            state_receipt_sha256=srda._canonical_sha256(payload),
        )


def test_ground_prior_memory_tamper_and_phase1_locked_swap_fail_closed() -> None:
    bank = _bank(5)
    prior = _ground(3)
    lock = _b_lock(5)
    prior.basis_codes_qint8.setflags(write=True)
    prior.basis_codes_qint8[0, 0] = np.int8(126)
    with pytest.raises(srda.ZIDSRDAError):
        srda.decode_shared_covariance_prior(prior)
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_zid_srda_state(bank, prior, lock=lock)

    swapped = srda.build_typed_shared_covariance_prior(
        np.eye(zid.Z_DIM, 3, dtype=np.float32),
        np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        provenance=srda.TypedZIDGroundPriorProvenance(
            source_receipt_sha256="d" * 64
        ),
    )
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_zid_srda_state(bank, swapped, lock=_b_lock(5))


def test_same_k_a_config_swap_and_unrelated_resource_pair_fail_closed() -> None:
    support, labels = _support(5)
    alternate_a = replace(_a_lock(5), temperature=1.25)
    alternate_bank = zid.build_typed_zid_support_bank(
        support, labels, CLASSES, config=alternate_a
    )
    with pytest.raises(srda.ZIDSRDAError):
        srda.build_zid_srda_state(
            alternate_bank, None, lock=_b_lock(5, ground_weight=0.0)
        )

    bank = _bank(5)
    state = srda.build_zid_srda_state(
        bank, None, lock=_b_lock(5, ground_weight=0.0)
    )
    with pytest.raises(srda.ZIDSRDAError):
        srda.audit_combined_resources(_bank(5, seed=999), state)
    with pytest.raises(srda.ZIDSRDAError):
        srda.audit_combined_resources(_bank(10), state)


def test_fusion_result_rejects_probability_range_and_prediction_argmax_drift() -> None:
    valid = np.asarray([[0.6, 0.3, 0.1]], dtype=np.float32)
    negative = np.asarray([[1.1, -0.1, 0.0]], dtype=np.float32)
    with pytest.raises(srda.ZIDSRDAError):
        srda.TypedZIDSRDAFusionResult(
            classes=CLASSES,
            a_probability_fp32=valid,
            rda_probability_fp32=None,
            fused_probability_fp32=negative,
            predicted_class_indices_int16=np.asarray([0], dtype=np.int16),
            audit={},
        )
    with pytest.raises(srda.ZIDSRDAError):
        srda.TypedZIDSRDAFusionResult(
            classes=CLASSES,
            a_probability_fp32=valid,
            rda_probability_fp32=None,
            fused_probability_fp32=valid,
            predicted_class_indices_int16=np.asarray([1], dtype=np.int16),
            audit={},
        )


def test_wire_nan_duplicate_shape_overflow_and_payload_bitflip_are_typed_errors() -> None:
    state = srda.build_zid_srda_state(
        _bank(5), None, lock=_b_lock(5, ground_weight=0.0)
    )
    wire = srda.serialize_zid_srda_state(state)
    prefix_size = len(srda.WIRE_MAGIC)
    header_start = prefix_size + 4
    header_size = struct.unpack("<I", wire[prefix_size:header_start])[0]
    header_raw = wire[header_start : header_start + header_size]
    suffix = wire[header_start + header_size :]

    def reframe(raw: bytes) -> bytes:
        return wire[:prefix_size] + struct.pack("<I", len(raw)) + raw + suffix

    header = json.loads(header_raw)
    header["alpha_phase1"] = float("nan")
    nan_raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(srda.ZIDSRDAError):
        srda.deserialize_zid_srda_state(reframe(nan_raw))

    duplicate_raw = b'{"schema":"duplicate",' + header_raw[1:]
    with pytest.raises(srda.ZIDSRDAError):
        srda.deserialize_zid_srda_state(reframe(duplicate_raw))

    array_start = header_start + header_size + 2
    name_size = struct.unpack("<H", wire[array_start : array_start + 2])[0]
    position = array_start + 2 + name_size
    dtype_size = struct.unpack("<H", wire[position : position + 2])[0]
    position += 2 + dtype_size
    assert wire[position] == 2
    first_dimension = position + 1
    overflow = bytearray(wire)
    overflow[first_dimension : first_dimension + 4] = struct.pack("<I", 0xFFFFFFFF)
    with pytest.raises(srda.ZIDSRDAError):
        srda.deserialize_zid_srda_state(bytes(overflow))

    payload_flip = bytearray(wire)
    payload_flip[-1] ^= 1
    with pytest.raises(srda.ZIDSRDAError):
        srda.deserialize_zid_srda_state(bytes(payload_flip))


def test_wire_resigned_numeric_type_aliases_fail_closed() -> None:
    state = srda.build_zid_srda_state(
        _bank(5), None, lock=_b_lock(5, alpha=1.0, ground_weight=0.0)
    )
    wire = srda.serialize_zid_srda_state(state)
    prefix_size = len(srda.WIRE_MAGIC)
    header_start = prefix_size + 4
    header_size = struct.unpack("<I", wire[prefix_size:header_start])[0]
    base_header = json.loads(wire[header_start : header_start + header_size])
    suffix = wire[header_start + header_size :]
    base_payload = srda._state_payload(
        classes=state.classes,
        active_k=state.active_k,
        codes=state.weight_codes_qint8,
        scales=state.weight_scales_fp16,
        bias=state.bias_fp16,
        alpha=state.alpha_phase1,
        a_bank_receipt_sha256=state.a_bank_receipt_sha256,
        a_config_lock_digest=state.a_config_lock_digest,
        a_metric_receipt_sha256=state.a_metric_receipt_sha256,
        ground_prior_receipt_sha256=state.ground_prior_receipt_sha256,
        lock=state.lock,
        fit_audit=state.fit_audit,
        quantization_audit=state.quantization_audit,
        resource_audit=state.resource_audit,
    )

    def attack(kind: str) -> bytes:
        header = copy.deepcopy(base_header)
        payload = copy.deepcopy(base_payload)
        if kind == "state_alpha_bool":
            header["alpha_phase1"] = True
            payload["alpha_phase1"] = True
        elif kind == "state_temperature_string":
            header["rda_temperature"] = "1.1"
        elif kind == "lock_sigma_string":
            header["lock"]["sigma0_sq"] = "0.7"
            payload["lock"]["sigma0_sq"] = "0.7"
            payload["lock_digest"] = srda._canonical_sha256(header["lock"])
        elif kind == "audit_shrinkage_string":
            header["fit_audit"]["shrinkage"] = str(
                header["fit_audit"]["shrinkage"]
            )
            payload["fit_audit"]["shrinkage"] = header["fit_audit"]["shrinkage"]
        else:
            raise AssertionError(kind)
        header["state_receipt_sha256"] = srda._canonical_sha256(payload)
        raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        return wire[:prefix_size] + struct.pack("<I", len(raw)) + raw + suffix

    for kind in (
        "state_alpha_bool",
        "state_temperature_string",
        "lock_sigma_string",
        "audit_shrinkage_string",
    ):
        with pytest.raises(srda.ZIDSRDAError):
            srda.deserialize_zid_srda_state(attack(kind))
