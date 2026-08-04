from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from cvsrffi import stage2_d129_joint6_heads as d129
from cvsrffi import stage2_next_r3_tsl160 as tsl


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _raw_cluster(class_index: int, count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.normal(0.02, 0.025, size=(count, 160)).astype(np.float32)
    value[:, class_index] += np.float32(4.0)
    value[:, 32 + class_index] += np.float32(0.7)
    return value


def _cell(receiver: str, label: str, class_index: int, *, seed: int) -> tsl.TSL160Phase1Cell:
    rows = _raw_cluster(class_index, 6, seed=seed)
    return tsl.TSL160Phase1Cell(
        receiver_id=receiver,
        class_handle=label,
        physical_ids=tuple(f"{receiver}-{label}-{index}" for index in range(len(rows))),
        zid160=rows,
    )


def _phase1_fixture() -> tuple[
    tuple[tsl.TSL160Phase1Cell, ...],
    tsl.TSL160PhysicalLOOFold,
    tsl.TSL160RuntimeBinding,
]:
    cells = [_cell("r0", "c0", 0, seed=10), _cell("r0", "c1", 1, seed=11), _cell("r1", "c0", 0, seed=12)]
    for receiver, offset in (("r1", 20), ("r2", 40)):
        for class_index in range(1, 5):
            cells.append(_cell(receiver, f"c{class_index}", class_index, seed=offset + class_index))
    cells = tuple(cells)
    class_cells = tuple(
        next(cell for cell in cells if cell.receiver_id == "r1" and cell.class_handle == f"c{class_index}")
        for class_index in range(1, 5)
    )
    fold = tsl.TSL160PhysicalLOOFold(
        fold_id="phase1-r1-c1-physical-loo",
        receiver_id="r1",
        class_handle="c1",
        registered_classes=("c1", "c2", "c3", "c4"),
        support_zid160=np.concatenate(tuple(cell.zid160[:5] for cell in class_cells), axis=0),
        support_labels=tuple(f"c{class_index}" for class_index in range(1, 5) for _ in range(5)),
        support_physical_ids=tuple(physical_id for cell in class_cells for physical_id in cell.physical_ids[:5]),
        validation_zid160=np.concatenate(tuple(cell.zid160[5:] for cell in class_cells), axis=0),
        validation_labels=("c1", "c2", "c3", "c4"),
        validation_physical_ids=tuple(physical_id for cell in class_cells for physical_id in cell.physical_ids[5:]),
    )
    eligible = tuple(
        cell for cell in cells if cell.receiver_id != "r0" and cell.class_handle != "c0"
    )
    binding = tsl.TSL160RuntimeBinding(
        outer_fold_id="outer-r0-c0",
        checkpoint_sha256=_sha("checkpoint"),
        representation_rule_sha256=_sha("d106-canonical"),
        phase1_physical_id_root_sha256=tsl.phase1_physical_id_root(eligible),
        phase1_seal_sha256=_sha("phase1-seal"),
    )
    return cells, fold, binding


def _prior() -> tuple[tsl.TSL160Phase1Prior, tsl.TSL160RuntimeBinding]:
    cells, fold, binding = _phase1_fixture()
    built = tsl.build_tsl160_phase1_prior(
        cells,
        (fold,),
        binding=binding,
        held_receiver="r0",
        held_class="c0",
    )
    return built.prior, binding


def _k5_support() -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    values = tuple(_raw_cluster(class_index, 5, seed=120 + class_index) for class_index in range(1, 5))
    return (
        np.concatenate(values, axis=0),
        tuple(f"c{class_index}" for class_index in range(1, 5) for _ in range(5)),
        ("c1", "c2", "c3", "c4"),
    )


def _r0_binding_kwargs(rows: np.ndarray) -> dict[str, str]:
    return {
        "representation_mode": tsl.CANONICAL_R0,
        "representation_context_sha256": _sha("r0-d106-cache-context"),
        "support_cache_sha256": tsl.tsl160_cache_sha256(rows),
    }


def _r1_binding_kwargs(rows: np.ndarray) -> dict[str, str]:
    return {
        "representation_mode": tsl.RDCE_R1_SIGNED_UNIT,
        "representation_context_sha256": _sha("sealed-rdce-bridge-context"),
        "support_cache_sha256": tsl.tsl160_cache_sha256(rows),
    }


def test_canonical_d106_is_relu_normalized_and_fails_on_zero() -> None:
    raw = np.full((1, 160), -1.0, dtype=np.float32)
    raw[0, 3] = np.float32(2.0)
    value = tsl.canonical_d106_relu_zid160(raw)
    assert value.dtype == np.float32
    assert not value.flags.writeable
    assert value[0, 3] == pytest.approx(1.0)
    assert float(np.min(value)) >= 0.0
    with pytest.raises(tsl.NextR3TSL160Error, match="zero"):
        tsl.canonical_d106_relu_zid160(np.zeros((1, 160), dtype=np.float32))


def test_phase1_prior_double_exclusion_binding_and_readonly_roundtrip() -> None:
    cells, fold, binding = _phase1_fixture()
    built = tsl.build_tsl160_phase1_prior(
        cells,
        (fold,),
        binding=binding,
        held_receiver="r0",
        held_class="c0",
    )
    prior = built.prior
    receipt = built.receipt
    assert prior.numeric_payload_bytes == 170
    assert not prior.q_logv0_int8.flags.writeable
    assert receipt["outer_fold_dual_exclusion"] is True
    assert receipt["binding_sha256"] == binding.binding_sha256
    assert receipt["excluded_held_receiver_cell_count"] == 2
    assert receipt["excluded_held_class_cell_count"] == 2
    wire = tsl.serialize_tsl160_prior(prior)
    recovered = tsl.roundtrip_tsl160_prior(prior)
    assert tsl.serialize_tsl160_prior(recovered) == wire
    assert recovered.prior_sha256 == prior.prior_sha256

    bad_fold = replace(fold, receiver_id="r0")
    with pytest.raises(tsl.NextR3TSL160Error, match="held receiver/class exclusion"):
        tsl.build_tsl160_phase1_prior(
            cells,
            (bad_fold,),
            binding=binding,
            held_receiver="r0",
            held_class="c0",
        )


def test_k1_is_exact_qknn_object_alias_and_tie_fails_closed() -> None:
    prior, binding = _prior()
    support = tsl.canonical_d106_relu_zid160(
        np.concatenate(tuple(_raw_cluster(class_index, 1, seed=130 + class_index) for class_index in range(1, 5)), axis=0)
    )
    fit = tsl.fit_tsl160(
        support,
        ("c1", "c2", "c3", "c4"),
        ("c1", "c2", "c3", "c4"),
        prior=prior,
        runtime_binding=binding,
        **_r0_binding_kwargs(support),
    )
    assert type(fit.state) is d129.D129K1QKNNAliasState
    assert fit.fit_receipt["fit_mode"] == "exact_qknn_logit_object_alias"
    assert fit.resource_receipt["incremental_deployed_numeric_state_bytes"] == 0
    logits = np.asarray([[4.0, 1.0, 0.0, -1.0], [0.0, 5.0, 1.0, -1.0]], dtype=np.float32)
    assert tsl.alias_k1_qknn_logits(fit, logits, runtime_binding=binding) is logits
    with pytest.raises(tsl.NextR3TSL160TieError):
        tsl.alias_k1_qknn_logits(
            fit,
            np.asarray([[1.0, 1.0, 0.0, -1.0]], dtype=np.float32),
            runtime_binding=binding,
        )


def test_k5_d129_wire_eb_resource_and_permutation_equivariance() -> None:
    prior, binding = _prior()
    support_raw, labels, classes = _k5_support()
    support = tsl.canonical_d106_relu_zid160(support_raw)
    fit = tsl.fit_tsl160(
        support,
        labels,
        classes,
        prior=prior,
        runtime_binding=binding,
        **_r0_binding_kwargs(support),
    )
    assert type(fit.state) is d129.D129AffineHeadState
    assert fit.state.head == d129.LITE_HEAD
    assert fit.state.numeric_state_bytes == 164 * len(classes)
    assert fit.fit_receipt["role_input"] is False
    assert fit.fit_receipt["same_formula_all_registered_classes"] is True
    assert fit.fit_receipt["query_rows_used_for_fit"] == 0
    assert fit.resource_receipt["fit_analytic_mac_formula"] == "4*N*160+8*160+2*C*160"
    assert fit.resource_receipt["explicit_dense_matrix_elements_constructed"] == 0
    assert fit.resource_receipt["explicit_spectral_factorization_count"] == 0
    assert fit.resource_receipt["explicit_linear_system_solve_count"] == 0
    tsl.validate_tsl160_fit_binding(fit, binding)

    query_raw = np.concatenate(tuple(_raw_cluster(class_index, 2, seed=140 + class_index) for class_index in range(1, 5)), axis=0)
    query = tsl.canonical_d106_relu_zid160(query_raw)
    expected = d129.score_d129_affine_head(fit.state, query)
    actual = tsl.score_tsl160_affine(
        fit,
        query,
        runtime_binding=binding,
        representation_mode=tsl.CANONICAL_R0,
        representation_context_sha256=_sha("r0-d106-cache-context"),
        query_cache_sha256=tsl.tsl160_cache_sha256(query),
    )
    np.testing.assert_array_equal(actual, expected)

    permutation = np.concatenate(tuple(support[index * 5 : (index + 1) * 5] for index in (3, 2, 1, 0)), axis=0)
    permuted = tsl.fit_tsl160(
        permutation,
        ("c4",) * 5 + ("c3",) * 5 + ("c2",) * 5 + ("c1",) * 5,
        ("c4", "c3", "c2", "c1"),
        prior=prior,
        runtime_binding=binding,
        **_r0_binding_kwargs(permutation),
    )
    permuted_logits = tsl.score_tsl160_affine(
        permuted,
        query,
        runtime_binding=binding,
        representation_mode=tsl.CANONICAL_R0,
        representation_context_sha256=_sha("r0-d106-cache-context"),
        query_cache_sha256=tsl.tsl160_cache_sha256(query),
    )
    np.testing.assert_allclose(actual, permuted_logits[:, ::-1], rtol=0.0, atol=2.0e-5)

    wrong_binding = replace(binding, checkpoint_sha256=_sha("other-checkpoint"))
    with pytest.raises(tsl.NextR3TSL160Error, match="does not match"):
        tsl.validate_tsl160_fit_binding(fit, wrong_binding)


def test_r1_signed_unit_cache_is_direct_bound_and_source_anchor_is_explicit() -> None:
    prior, binding = _prior()
    support_raw, labels, classes = _k5_support()
    r1_support = tsl.canonical_d106_relu_zid160(support_raw).copy()
    r1_support[:, :80] *= np.float32(-1.0)
    support_before = r1_support.copy()
    fit = tsl.fit_tsl160(
        r1_support,
        labels,
        classes,
        prior=prior,
        runtime_binding=binding,
        **_r1_binding_kwargs(r1_support),
    )
    assert np.any(r1_support < 0.0)
    np.testing.assert_array_equal(r1_support, support_before)
    assert fit.fit_receipt["representation_mode"] == tsl.RDCE_R1_SIGNED_UNIT
    assert fit.fit_receipt["support_cache_sha256"] == tsl.tsl160_cache_sha256(r1_support)
    assert fit.fit_receipt["representation_context_sha256"] == _sha("sealed-rdce-bridge-context")
    assert fit.fit_receipt["prior_semantics"] == "pre_adaptation_source_anchor_same_ambient_axes"
    assert fit.fit_receipt["prior_transported_by_rdce"] is False
    assert fit.fit_receipt["r1_covariance_claim"] is False
    with pytest.raises(tsl.NextR3TSL160Error, match="non-negative D106 ReLU cache"):
        tsl.fit_tsl160(
            r1_support,
            labels,
            classes,
            prior=prior,
            runtime_binding=binding,
            **_r0_binding_kwargs(r1_support),
        )

    query_raw = np.concatenate(tuple(_raw_cluster(class_index, 2, seed=200 + class_index) for class_index in range(1, 5)), axis=0)
    r1_query = tsl.canonical_d106_relu_zid160(query_raw).copy()
    r1_query[:, :80] *= np.float32(-1.0)
    query_before = r1_query.copy()
    expected = d129.score_d129_affine_head(fit.state, r1_query)
    actual = tsl.score_tsl160_affine(
        fit,
        r1_query,
        runtime_binding=binding,
        representation_mode=tsl.RDCE_R1_SIGNED_UNIT,
        representation_context_sha256=_sha("sealed-rdce-bridge-context"),
        query_cache_sha256=tsl.tsl160_cache_sha256(r1_query),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(r1_query, query_before)

    with pytest.raises(tsl.NextR3TSL160Error, match="representation/context"):
        tsl.score_tsl160_affine(
            fit,
            r1_query,
            runtime_binding=binding,
            representation_mode=tsl.RDCE_R1_SIGNED_UNIT,
            representation_context_sha256=_sha("different-rdce-context"),
            query_cache_sha256=tsl.tsl160_cache_sha256(r1_query),
        )
    with pytest.raises(tsl.NextR3TSL160Error, match="cache SHA256 drift"):
        tsl.score_tsl160_affine(
            fit,
            r1_query,
            runtime_binding=binding,
            representation_mode=tsl.RDCE_R1_SIGNED_UNIT,
            representation_context_sha256=_sha("sealed-rdce-bridge-context"),
            query_cache_sha256=_sha("drifted-r1-query-cache"),
        )
