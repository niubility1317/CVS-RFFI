"""Focused D108 D92 core tests and compact in-file traceability.

| ID | Requirement | Verification |
|---|---|---|
| D108-CORE-01 | Fixed candidate identity and minimal formal-int8 pair | identity/state-field test |
| D108-CORE-02 | Strict temporary D92 injection and recovery | injection/exception tests |
| D108-CORE-03 | Fixed four-arm scoring and exact M0 path | direct D42/D92 equality test |
| D108-CORE-04 | K1, singleton decisions, and class permutation | K1/per-row/permutation tests |
| D108-CORE-05 | Support-only capability and resource closure | surface/resource tests |
"""

from __future__ import annotations

from dataclasses import fields
import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
import cvsrffi.stage2_d108_d92_core as core
import cvsrffi.stage2_d108_smme as smme
from cvsrffi.stage2_d92_registration_balanced_covariance import (
    build_registration_balanced_equal_lda,
)


OLD_CLASSES = tuple(f"old_{index}" for index in range(6))
NEW_CLASSES = tuple(f"new_{index}" for index in range(5))
ALL_CLASSES = OLD_CLASSES + NEW_CLASSES


def _one_support_row(class_index: int, shot: int, k_shot: int) -> np.ndarray:
    rng = np.random.default_rng(108_000 + 101 * class_index + shot)
    row = np.empty(d42.FEATURE_DIM, dtype=np.float32)
    relu_center = np.full(160, 0.12 + 0.011 * class_index, dtype=np.float32)
    relu_center[(11 * class_index) % 160] += np.float32(1.2)
    relu_center[(17 * class_index + 7) % 160] += np.float32(0.45)
    row[:160] = np.maximum(
        relu_center + np.asarray(rng.normal(0.0, 0.055, size=160), dtype=np.float32),
        np.float32(0.0),
    )
    spectral_center = np.float32(0.09 * (class_index - 5))
    row[160:256] = spectral_center + np.asarray(
        rng.normal(0.0, 0.17, size=96), dtype=np.float32
    )
    rf_center = np.float32(-0.06 * (class_index - 4))
    row[256:] = rf_center + np.asarray(
        rng.normal(0.0, 0.14, size=32), dtype=np.float32
    )
    if k_shot == 1:
        # Keep K1 deterministic while retaining class-distinct nonzero geometry.
        row[:160] += np.float32(0.003 * class_index)
    return row


def _support(k_shot: int) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    old_rows: list[np.ndarray] = []
    old_labels: list[str] = []
    new_rows: list[np.ndarray] = []
    new_labels: list[str] = []
    for class_index, class_name in enumerate(ALL_CLASSES):
        for shot in range(k_shot):
            row = _one_support_row(class_index, shot, k_shot)
            if class_name in OLD_CLASSES:
                old_rows.append(row)
                old_labels.append(class_name)
            else:
                new_rows.append(row)
                new_labels.append(class_name)
    old_order = list(reversed(range(len(old_rows))))
    new_order = list(reversed(range(len(new_rows))))
    return (
        np.stack([old_rows[index] for index in old_order]).astype(np.float32),
        [old_labels[index] for index in old_order],
        np.stack([new_rows[index] for index in new_order]).astype(np.float32),
        [new_labels[index] for index in new_order],
    )


@pytest.fixture
def fast_d42_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep core tests focused on pair wiring rather than B20 optimization cost."""

    def _fit(
        old_rows: np.ndarray,
        old_targets: np.ndarray,
        old_class_count: int,
        *,
        seed: int,
        device: d42.torch.device,
    ) -> tuple[np.ndarray, tuple[dict[str, object], ...], dict[str, object]]:
        del old_targets, seed, device
        trace = tuple(
            {"optimizer_step": step} for step in range(1, d42.METRIC_EPOCHS + 1)
        )
        return (
            np.zeros(d42.FEATURE_DIM, dtype=np.float32),
            trace,
            {
                "trainable_parameters": d42.FEATURE_DIM * (1 + old_class_count),
                "optimizer_steps": d42.METRIC_EPOCHS,
                "estimated_adaptation_macs": len(old_rows)
                * old_class_count
                * d42.FEATURE_DIM,
                "peak_cuda_memory_bytes": 0,
            },
        )

    monkeypatch.setattr(d42, "_fit_old_only_b3_metric", _fit)


def _d92_fit():
    return build_registration_balanced_equal_lda(
        d42, d42._fit_equal_prior_lda, arm="full"
    )


def _build(k_shot: int = 5) -> tuple[core.D108D92Pair, tuple[np.ndarray, ...], tuple[list[str], ...]]:
    old_x, old_y, new_x, new_y = _support(k_shot)
    pair = core.build_d108_d92_pair(
        old_x,
        old_y,
        OLD_CLASSES,
        new_x,
        new_y,
        NEW_CLASSES,
        seed=108,
        device="cpu",
        d92_fit=_d92_fit(),
    )
    return pair, (old_x, new_x), (old_y, new_y)


def _reorder_support(
    rows: np.ndarray, labels: list[str], classes: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    indices = [index for class_name in classes for index, label in enumerate(labels) if label == class_name]
    return rows[indices].copy(), [labels[index] for index in indices]


def test_candidate_identity_and_pair_keep_only_formal_int8_score_states(
    fast_d42_metric: None,
) -> None:
    pair, _features, _labels = _build(5)
    assert core.CANDIDATE_ID == "D108-CB-RRC-SMME/r1"
    pair_fields = {field.name for field in fields(core.D108D92Pair)}
    assert {"base_result", "da_result", "matched_fp32_before_state", "matched_fp32_state"}.isdisjoint(pair_fields)
    assert {
        "base_before_state",
        "base_after_state",
        "da_before_state",
        "da_after_state",
    }.issubset(pair_fields)
    for state in (
        pair.base_before_state,
        pair.base_after_state,
        pair.da_before_state,
        pair.da_after_state,
    ):
        assert state.is_int8
        assert state.coef_fp32.shape == (0, d42.FEATURE_DIM)
        assert state.intercept_fp32.shape == (0,)
        assert not state.coef1_qint8.flags.writeable
    assert pair.base_before_state.classes == OLD_CLASSES
    assert pair.base_after_state.classes == ALL_CLASSES


def test_strict_d92_injection_builds_base_and_da_then_restores_global(
    fast_d42_metric: None,
) -> None:
    old_x, old_y, new_x, new_y = _support(5)
    original_fit = d42._fit_equal_prior_lda
    frozen_d92 = _d92_fit()
    calls: list[tuple[int, int]] = []

    def traced_d92(*args):
        assert d42._fit_equal_prior_lda is traced_d92
        calls.append((int(args[2]), int(args[3])))
        return frozen_d92(*args)

    pair = core.build_d108_d92_pair(
        old_x,
        old_y,
        OLD_CLASSES,
        new_x,
        new_y,
        NEW_CLASSES,
        seed=108,
        device="cpu",
        d92_fit=traced_d92,
    )
    assert isinstance(pair, core.D108D92Pair)
    assert calls == [(6, 5), (11, 5), (6, 5), (11, 5)]
    assert d42._fit_equal_prior_lda is original_fit


def test_exception_during_d92_fit_restores_d42_global(
    fast_d42_metric: None,
) -> None:
    old_x, old_y, new_x, new_y = _support(1)
    original_fit = d42._fit_equal_prior_lda

    def exploding_d92(*_args):
        assert d42._fit_equal_prior_lda is exploding_d92
        raise RuntimeError("injected D92 failure")

    with pytest.raises(RuntimeError, match="injected D92 failure"):
        core.build_d108_d92_pair(
            old_x,
            old_y,
            OLD_CLASSES,
            new_x,
            new_y,
            NEW_CLASSES,
            seed=108,
            device="cpu",
            d92_fit=exploding_d92,
        )
    assert d42._fit_equal_prior_lda is original_fit


def test_m0_is_bitwise_direct_d42_d92_score_and_heads_use_formal_support_logits(
    fast_d42_metric: None,
) -> None:
    pair, (old_x, new_x), (old_y, new_y) = _build(5)
    all_x = np.concatenate((old_x, new_x), axis=0).astype(np.float32)
    all_y = old_y + new_y
    for phase, state, support_x, support_y, classes in (
        ("before", pair.base_before_state, old_x, old_y, OLD_CLASSES),
        ("after", pair.base_after_state, all_x, all_y, ALL_CLASSES),
    ):
        query = support_x[:3].copy()
        direct = d42.score_d42_unified_shrinkage_lda(state, query)
        actual = core.score(pair, phase, "M0", query)
        np.testing.assert_array_equal(actual, direct)
        rebuilt_smme = smme.build_smme_state(
            d42.score_d42_unified_shrinkage_lda(state, support_x), support_y, classes
        )
        selected_smme = pair.base_before_smme if phase == "before" else pair.base_after_smme
        assert selected_smme.state_receipt_sha256 == rebuilt_smme.state_receipt_sha256
        np.testing.assert_array_equal(
            core.score(pair, phase, "M_HEAD", query),
            smme.apply_smme_query(rebuilt_smme, direct),
        )


def test_k1_all_four_arms_are_active_and_query_rows_are_independent(
    fast_d42_metric: None,
) -> None:
    pair, (old_x, new_x), _labels = _build(1)
    assert pair.k_shot == 1
    assert np.any(pair.base_before_smme.delta_fp64 != 0.0)
    assert np.any(pair.da_before_smme.delta_fp64 != 0.0)
    for phase, query in (("before", old_x[:3]), ("after", np.concatenate((old_x, new_x))[:3])):
        outputs = {arm: core.score(pair, phase, arm, query) for arm in core.ARMS}
        assert all(value.shape == (len(query), len(OLD_CLASSES) if phase == "before" else len(ALL_CLASSES)) for value in outputs.values())
        assert not np.array_equal(outputs["M0"], outputs["M_HEAD"])
        assert not np.array_equal(outputs["M_DA"], outputs["M_JOINT"])
        for arm, batch in outputs.items():
            singleton = np.concatenate(
                [core.score(pair, phase, arm, query[index : index + 1]) for index in range(len(query))],
                axis=0,
            )
            np.testing.assert_array_equal(batch, singleton)


def test_class_permutation_equivariance_and_before_state_stability(
    fast_d42_metric: None,
) -> None:
    pair, (old_x, new_x), (old_y, new_y) = _build(5)
    old_permutation = ("old_3", "old_0", "old_5", "old_1", "old_4", "old_2")
    new_permutation = ("new_4", "new_1", "new_3", "new_0", "new_2")
    perm_old_x, perm_old_y = _reorder_support(old_x, old_y, old_permutation)
    perm_new_x, perm_new_y = _reorder_support(new_x, new_y, new_permutation)
    before_snapshot = d42._state_snapshot(pair.base_before_state)
    permuted = core.build_d108_d92_pair(
        perm_old_x,
        perm_old_y,
        old_permutation,
        perm_new_x,
        perm_new_y,
        new_permutation,
        seed=108,
        device="cpu",
        d92_fit=_d92_fit(),
    )
    assert d42._state_snapshot(pair.base_before_state) == before_snapshot
    query = np.concatenate((old_x, new_x), axis=0)[:4]
    permuted_classes = old_permutation + new_permutation
    source_columns = [ALL_CLASSES.index(class_name) for class_name in permuted_classes]
    for arm in core.ARMS:
        original_scores = core.score(pair, "after", arm, query)
        permuted_scores = core.score(permuted, "after", arm, query)
        np.testing.assert_allclose(
            permuted_scores,
            original_scores[:, source_columns],
            rtol=0.0,
            atol=3.0e-4,
        )


def test_public_surfaces_exclude_truth_role_quota_and_build_has_no_query(
    fast_d42_metric: None,
) -> None:
    build_parameters = set(inspect.signature(core.build_d108_d92_pair).parameters)
    score_parameters = set(inspect.signature(core.score).parameters)
    forbidden = ("truth", "role", "quota", "batch_class_count", "update")
    assert "query" not in build_parameters
    assert score_parameters == {"pair", "phase", "arm", "query_features"}
    assert not any(
        token in parameter
        for parameter in build_parameters | score_parameters
        for token in forbidden
    )
    assert not any(
        token in field.name
        for field in fields(core.D108D92Pair)
        for token in ("truth", "role", "quota", "query")
    )
    pair, _features, _labels = _build(1)
    with pytest.raises(core.D108D92CoreError):
        core.score(pair, "unknown", "M0", np.zeros((1, 288), dtype=np.float32))
    with pytest.raises(core.D108D92CoreError):
        core.score(pair, "before", "unknown", np.zeros((1, 288), dtype=np.float32))


def test_resource_summary_is_closed_to_formal_states_and_support_only(
    fast_d42_metric: None,
) -> None:
    pair, _features, _labels = _build(5)
    summary = core.resource_summary(pair)
    assert summary["candidate_id"] == "D108-CB-RRC-SMME/r1"
    assert summary["formal_int8_score_state_count"] == 4
    assert summary["formal_int8_score_states_only"] is True
    assert summary["formal_score_state_bytes"] > 0
    assert summary["d42_trainable_parameters_per_fit"] == d42.FEATURE_DIM * 7
    assert summary["aggregate_d42_optimizer_steps"] == 2 * d42.METRIC_EPOCHS
    assert summary["support_only"] is True
    assert summary["cbrrc_state_source_old_before_support_only"] is True
    assert summary["query_rows_used_for_fit"] == 0
    assert summary["query_state_updates"] == 0
    assert summary["query_truth_access"] is False
    assert summary["query_role_access"] is False
    assert summary["query_class_quota_access"] is False
    assert summary["query_batch_global_assignment"] is False
    assert summary["clean_sample_access"] is False
    assert summary["source_sample_access"] is False
