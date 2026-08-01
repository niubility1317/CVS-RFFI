"""Focused D109 four-arm D92/CB-RRC integration tests."""

from __future__ import annotations

from dataclasses import fields, replace
import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi import stage2_d108_d92_core as d108
from cvsrffi import stage2_d109_d92_core as d109
from cvsrffi import stage2_d109_scrc as scrc


OLD_CLASSES = tuple(f"old-{index}" for index in range(6))
NEW_CLASSES = tuple(f"new-{index}" for index in range(5))
_D42_BASE_FIT = d42._fit_equal_prior_lda


def _support(k_shot: int = 1) -> tuple[
    np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]
]:
    rng = np.random.default_rng(71_309 + k_shot)

    def rows_for(classes: tuple[str, ...], offset: float) -> tuple[np.ndarray, tuple[str, ...]]:
        rows: list[np.ndarray] = []
        labels: list[str] = []
        for class_index, class_name in enumerate(classes):
            center = rng.uniform(0.05, 0.90, size=288).astype(np.float32)
            center[:160] += np.float32(offset + 0.08 * class_index)
            for _ in range(k_shot):
                row = center + rng.normal(0.0, 0.002, size=288).astype(np.float32)
                row[:160] = np.maximum(row[:160], np.float32(0.0))
                rows.append(row)
                labels.append(class_name)
        return np.ascontiguousarray(np.stack(rows), dtype=np.float32), tuple(labels)

    old_rows, old_labels = rows_for(OLD_CLASSES, 0.02)
    new_rows, new_labels = rows_for(NEW_CLASSES, 0.31)
    return old_rows, old_labels, new_rows, new_labels


def _d92_fit(*args, **kwargs):  # type: ignore[no-untyped-def]
    return _D42_BASE_FIT(*args, **kwargs)


def _build(k_shot: int = 1) -> tuple[d109.D109D92Pair, tuple[np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]]]:
    payload = _support(k_shot)
    old_rows, old_labels, new_rows, new_labels = payload
    pair = d109.build_d109_d92_pair(
        old_rows,
        old_labels,
        OLD_CLASSES,
        new_rows,
        new_labels,
        NEW_CLASSES,
        seed=713_109,
        device="cpu",
        d92_fit=_d92_fit,
    )
    return pair, payload


def _query(old_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate((old_rows[:2], new_rows[:3]), axis=0), dtype=np.float32
    )


def _d42_snapshots(pair: d109.D109D92Pair) -> tuple[tuple[bytes, ...], ...]:
    return tuple(
        d42._state_snapshot(state)
        for state in (
            pair.base_before_state,
            pair.base_after_state,
            pair.da_before_state,
            pair.da_after_state,
        )
    )


def test_d109_reuses_exact_d108_m0_and_da_logits_without_persisting_smme() -> None:
    old_rows, old_labels, new_rows, new_labels = _support()
    d108_pair = d108.build_d108_d92_pair(
        old_rows,
        old_labels,
        OLD_CLASSES,
        new_rows,
        new_labels,
        NEW_CLASSES,
        seed=713_109,
        device="cpu",
        d92_fit=_d92_fit,
    )
    pair = d109.build_d109_d92_pair(
        old_rows,
        old_labels,
        OLD_CLASSES,
        new_rows,
        new_labels,
        NEW_CLASSES,
        seed=713_109,
        device="cpu",
        d92_fit=_d92_fit,
    )
    query = _query(old_rows, new_rows)
    for phase in d109.PHASES:
        np.testing.assert_array_equal(
            d109.score(pair, phase, "M0", query),
            d108.score(d108_pair, phase, "M0", query),
        )
        np.testing.assert_array_equal(
            d109.score(pair, phase, "M_DA", query),
            d108.score(d108_pair, phase, "M_DA", query),
        )
    assert not any("smme" in field.name.lower() for field in fields(d109.D109D92Pair))
    assert tuple(type(state) for state in (
        pair.base_before_scrc,
        pair.base_after_scrc,
        pair.da_before_scrc,
        pair.da_after_scrc,
    )) == (scrc.SCRCState,) * 4


def test_k1_support_only_per_query_and_resource_state_closure() -> None:
    pair, (old_rows, old_labels, new_rows, new_labels) = _build(k_shot=1)
    query = _query(old_rows, new_rows)
    before_snapshots = _d42_snapshots(pair)
    assert "query" not in inspect.signature(d109.build_d109_d92_pair).parameters
    with pytest.raises(TypeError):
        d109.build_d109_d92_pair(  # type: ignore[call-arg]
            old_rows,
            old_labels,
            OLD_CLASSES,
            new_rows,
            new_labels,
            NEW_CLASSES,
            seed=713_109,
            device="cpu",
            d92_fit=_d92_fit,
            query_features=query,
        )
    for phase in d109.PHASES:
        for arm in d109.ARMS:
            first = d109.score(pair, phase, arm, query)
            second = d109.score(pair, phase, arm, query)
            assert first.dtype == np.float32
            assert first.shape == (len(query), 6 if phase == "before" else 11)
            np.testing.assert_array_equal(first, second)
    assert _d42_snapshots(pair) == before_snapshots
    summary = d109.resource_summary(pair)
    assert summary["formal_int8_score_state_count"] == 4
    assert summary["scrc_state_count"] == 4
    assert summary["d108_smme_state_persisted"] is False
    assert summary["support_only"] is True
    assert summary["query_rows_used_for_fit"] == 0
    assert summary["query_state_updates"] == 0
    assert summary["query_truth_access"] is False
    assert summary["query_role_access"] is False
    assert summary["query_class_quota_access"] is False
    assert summary["query_batch_global_assignment"] is False
    assert summary["formal_score_state_bytes"] > summary["scrc_numeric_state_bytes"] > 0
    with pytest.raises(d109.D109D92CoreError, match="closure"):
        replace(pair, k_shot=5)


def test_class_permutation_equivariance_for_each_phase_and_arm() -> None:
    old_rows, old_labels, new_rows, new_labels = _support()
    original = d109.build_d109_d92_pair(
        old_rows,
        old_labels,
        OLD_CLASSES,
        new_rows,
        new_labels,
        NEW_CLASSES,
        seed=713_109,
        device="cpu",
        d92_fit=_d92_fit,
    )
    permuted_old = tuple(reversed(OLD_CLASSES))
    permuted_new = tuple(reversed(NEW_CLASSES))
    permuted = d109.build_d109_d92_pair(
        old_rows,
        old_labels,
        permuted_old,
        new_rows,
        new_labels,
        permuted_new,
        seed=713_109,
        device="cpu",
        d92_fit=_d92_fit,
    )
    query = _query(old_rows, new_rows)
    for phase in d109.PHASES:
        expected_classes = OLD_CLASSES if phase == "before" else OLD_CLASSES + NEW_CLASSES
        permuted_classes = permuted_old if phase == "before" else permuted_old + permuted_new
        restore_columns = [permuted_classes.index(class_name) for class_name in expected_classes]
        for arm in d109.ARMS:
            expected = d109.score(original, phase, arm, query)
            actual = d109.score(permuted, phase, arm, query)[:, restore_columns]
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_d108_global_fit_hook_recovers_after_d109_construction_error() -> None:
    old_rows, old_labels, new_rows, new_labels = _support()
    original_fit = d42._fit_equal_prior_lda
    original_runtime = d42.SKLEARN_RUNTIME_VERSION

    def fail_fit(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected D109 D92-fit failure")

    with pytest.raises(RuntimeError, match="injected D109 D92-fit failure"):
        d109.build_d109_d92_pair(
            old_rows,
            old_labels,
            OLD_CLASSES,
            new_rows,
            new_labels,
            NEW_CLASSES,
            seed=713_109,
            device="cpu",
            d92_fit=fail_fit,
        )
    assert d42._fit_equal_prior_lda is original_fit
    assert d42.SKLEARN_RUNTIME_VERSION == original_runtime
