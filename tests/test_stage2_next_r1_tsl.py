from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_next_r1_tsl import (
    Phase1Cell,
    Phase1PhysicalLOOFold,
    TSLAffineHeadState,
    TSLK1AliasState,
    TSLRuntimeBinding,
    TSLTieUnresolvedError,
    TailSafeLite,
    TailSafeLiteError,
    alias_qknn_logits,
    build_phase1_prior,
    deserialize_affine_head,
    deserialize_phase1_prior,
    normalize_signed_prerelu160,
    phase1_cell_physical_id_root,
    require_unique_float32_top,
    score_affine,
    serialize_affine_head,
    serialize_phase1_prior,
)


def _unit(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    rows /= np.sqrt(np.sum(rows * rows, axis=1, keepdims=True))
    return np.asarray(rows, dtype=np.float32)


def _rows(seed: int, center: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.normal(scale=0.04, size=(count, 160))
    rows[:, center] += 1.0
    rows[:, (center + 17) % 160] += np.linspace(-0.12, 0.12, count)
    return _unit(rows)


def _inputs():
    cells = []
    data = {}
    for receiver_index, receiver in enumerate(("rx0", "rx1")):
        for class_index, label in enumerate(("alpha", "beta")):
            rows = _rows(100 + 10 * receiver_index + class_index, class_index, 7)
            ids = tuple(f"{receiver}-{label}-{index:02d}" for index in range(7))
            cells.append(Phase1Cell(receiver=receiver, class_label=label, physical_ids=ids, z160=rows))
            data[(receiver, label)] = (rows, ids)
    alpha, alpha_ids = data[("rx0", "alpha")]
    beta, beta_ids = data[("rx0", "beta")]
    support = np.concatenate([alpha[:5], beta[:5]], axis=0)
    validation = np.concatenate([alpha[5:], beta[5:]], axis=0)
    fold = Phase1PhysicalLOOFold(
        fold_id="rx0-physical-loo",
        support_z160=support,
        support_labels=("alpha",) * 5 + ("beta",) * 5,
        registered_classes=("alpha", "beta"),
        support_physical_ids=alpha_ids[:5] + beta_ids[:5],
        validation_z160=validation,
        validation_labels=("alpha", "alpha", "beta", "beta"),
        validation_physical_ids=alpha_ids[5:] + beta_ids[5:],
    )
    build = build_phase1_prior(
        cells=cells,
        validation_folds=(fold,),
        checkpoint_sha256="a" * 64,
        cell_physical_id_root_sha256=phase1_cell_physical_id_root(cells),
        representation_rule_sha256="c" * 64,
    )
    return build, support, ("alpha",) * 5 + ("beta",) * 5, validation


def _binding(build, *, checkpoint_sha256="a" * 64, representation_rule_sha256="c" * 64, phase1_seal_sha256="d" * 64):
    return TSLRuntimeBinding(
        checkpoint_sha256=checkpoint_sha256,
        representation_rule_sha256=representation_rule_sha256,
        phase1_seal_sha256=phase1_seal_sha256,
    )


def test_signed_prerelu_totalization_and_exact_zero_failure():
    value = np.zeros((3, 160), dtype=np.float32)
    value[0, 0] = 3.0
    value[0, 1] = -2.0
    value[1, :2] = (-3.0, -4.0)
    result = normalize_signed_prerelu160(value[:2])
    assert np.isclose(np.linalg.norm(result[0]), 1.0)
    assert result[0, 1] == 0.0
    assert np.allclose(result[1, :2], np.asarray((-0.6, -0.8), dtype=np.float32))
    with pytest.raises(TailSafeLiteError, match="exact-zero"):
        normalize_signed_prerelu160(value[2:])


def test_phase1_prior_wire_is_deterministic_positive_and_downward_rho():
    build, _support, _labels, _validation = _inputs()
    wire = serialize_phase1_prior(build.prior)
    restored = deserialize_phase1_prior(wire)
    assert restored.prior_sha256 == build.prior.prior_sha256
    assert restored.serialized_bytes == wire
    assert np.all(restored.decoded_v0 > 0.0)
    assert build.receipt["rho_h_not_rounded_up"] is True
    assert build.receipt["physical_loo_margin_receipts"]
    assert all(item["support_validation_physical_ids_disjoint"] for item in build.receipt["physical_loo_margin_receipts"])


def test_phase1_prior_rejects_an_unbound_cell_id_root():
    build, support, labels, validation = _inputs()
    del build
    cells = [
        Phase1Cell("rx0", "alpha", tuple(f"a-{index}" for index in range(7)), _rows(21, 0, 7)),
        Phase1Cell("rx0", "beta", tuple(f"b-{index}" for index in range(7)), _rows(22, 1, 7)),
    ]
    fold = Phase1PhysicalLOOFold(
        fold_id="bound-root",
        support_z160=support,
        support_labels=labels,
        registered_classes=("alpha", "beta"),
        support_physical_ids=tuple(f"a-{index}" for index in range(5)) + tuple(f"b-{index}" for index in range(5)),
        validation_z160=validation,
        validation_labels=("alpha", "alpha", "beta", "beta"),
        validation_physical_ids=("a-5", "a-6", "b-5", "b-6"),
    )
    with pytest.raises(TailSafeLiteError, match="physical-ID root"):
        build_phase1_prior(
            cells=cells,
            validation_folds=(fold,),
            checkpoint_sha256="a" * 64,
            cell_physical_id_root_sha256="b" * 64,
            representation_rule_sha256="c" * 64,
        )


def test_runtime_binding_rejects_checkpoint_representation_and_row_seal_swaps():
    build, support, labels, validation = _inputs()
    with pytest.raises(TailSafeLiteError, match="checkpoint/representation"):
        TailSafeLite(build.prior, runtime_binding=_binding(build, checkpoint_sha256="e" * 64))
    with pytest.raises(TailSafeLiteError, match="checkpoint/representation"):
        TailSafeLite(build.prior, runtime_binding=_binding(build, representation_rule_sha256="f" * 64))
    binding = _binding(build)
    fitter = TailSafeLite(build.prior, runtime_binding=binding)
    fit = fitter.fit(support, labels, ("alpha", "beta"))
    swapped = _binding(build, phase1_seal_sha256="9" * 64)
    with pytest.raises(TailSafeLiteError, match="runtime binding mismatch"):
        score_affine(fit.state, validation, runtime_binding=swapped)


def test_physical_loo_rejects_overlap():
    rows = _rows(1, 0, 6)
    with pytest.raises(TailSafeLiteError, match="disjoint"):
        Phase1PhysicalLOOFold(
            fold_id="bad",
            support_z160=np.concatenate([rows[:5], rows[:5]], axis=0),
            support_labels=("alpha",) * 5 + ("beta",) * 5,
            registered_classes=("alpha", "beta"),
            support_physical_ids=tuple(f"s-{index}" for index in range(10)),
            validation_z160=rows[5:],
            validation_labels=("alpha",),
            validation_physical_ids=("s-0",),
        )


def test_k1_alias_is_exact_and_fit_api_has_no_role_or_query():
    build, _support, _labels, _validation = _inputs()
    binding = _binding(build)
    fitter = TailSafeLite(build.prior, runtime_binding=binding)
    signature = inspect.signature(TailSafeLite.fit)
    assert tuple(signature.parameters) == ("self", "support_z160", "support_labels", "registered_classes")
    rows = np.concatenate([_rows(901, 0, 1), _rows(902, 1, 1)], axis=0)
    fit = fitter.fit(rows, ("alpha", "beta"), ("alpha", "beta"))
    assert type(fit.state) is TSLK1AliasState
    logits = np.asarray([[0.7, 0.1], [0.1, 0.7]], dtype=np.float32)
    assert alias_qknn_logits(fit.state, logits, runtime_binding=binding) is logits
    with pytest.raises(TSLTieUnresolvedError):
        alias_qknn_logits(fit.state, np.asarray([[0.5, 0.5]], dtype=np.float32), runtime_binding=binding)
    with pytest.raises(TypeError):
        fitter.fit(rows, ("alpha", "beta"), ("alpha", "beta"), old_count=1)


def test_k5_fit_is_class_permutation_equivariant_and_reports_resources():
    build, support, labels, validation = _inputs()
    binding = _binding(build)
    fitter = TailSafeLite(build.prior, runtime_binding=binding)
    fit = fitter.fit(support, labels, ("alpha", "beta"))
    assert type(fit.state) is TSLAffineHeadState
    assert fit.state.numeric_state_bytes == 164 * 2
    assert fit.resource_receipt["query_head_macs_per_sample"] == 160 * 2
    assert fit.resource_receipt["head_fit_analytic_mac_equivalent"] == 4 * 10 * 160 + 8 * 160 + 2 * 2 * 160
    assert fit.resource_receipt["head_fit_wall_clock_ns"] > 0
    assert fit.resource_receipt["fabr_forward_cost_included"] is False
    assert fit.fit_receipt["old_new_role_access"] is False
    assert fit.fit_receipt["full_head_access"] is False
    assert fit.fit_receipt["support_compactness_policy"].startswith("physical_LOO")
    assert fit.fit_receipt["deployed_delta_from_scaled_reference"] > fit.fit_receipt["deployed_delta_tolerance"]
    base_scores = score_affine(fit.state, validation, runtime_binding=binding).logits
    permuted = fitter.fit(support, labels, ("beta", "alpha"))
    permuted_scores = score_affine(permuted.state, validation, runtime_binding=binding).logits
    assert np.allclose(base_scores, permuted_scores[:, ::-1], atol=2.0e-5, rtol=0.0)


def test_affine_head_wire_roundtrip_is_canonical_and_exact():
    build, support, labels, _validation = _inputs()
    fit = TailSafeLite(build.prior, runtime_binding=_binding(build)).fit(support, labels, ("alpha", "beta"))
    assert type(fit.state) is TSLAffineHeadState
    wire = serialize_affine_head(fit.state)
    restored = deserialize_affine_head(wire)
    assert restored.state_sha256 == fit.state.state_sha256
    assert restored.serialized_bytes == wire
    assert np.array_equal(restored.weight_qint8, fit.state.weight_qint8)
    assert np.array_equal(restored.scale_fp16, fit.state.scale_fp16)
    assert np.array_equal(restored.intercept_fp16, fit.state.intercept_fp16)


def test_top_tie_is_fail_closed_without_registry_or_hash_key():
    require_unique_float32_top(np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32))
    with pytest.raises(TSLTieUnresolvedError, match="TIE_UNRESOLVED"):
        require_unique_float32_top(np.asarray([[0.8, 0.8]], dtype=np.float32))
