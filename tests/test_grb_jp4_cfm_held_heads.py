from __future__ import annotations

import numpy as np
import pytest

from cvsrffi.grb_jp4_cfm_held_heads import (
    GRBJP4HeldHeadError,
    d92_resource_receipt,
    fit_held_d92_head,
    score_held_d92_head,
)


OLD = tuple(f"o{index}" for index in range(5))
NEW = ("n0",)


def _support(k: int):
    generator = np.random.default_rng(19 + k)
    labels = tuple(value for value in OLD + NEW for _ in range(k))
    centers = {
        value: generator.normal(size=160) for value in OLD + NEW
    }
    rows = np.stack(
        [centers[label] + 0.05 * generator.normal(size=160) for label in labels]
    ).astype(np.float32)
    return rows, labels


def test_k5_d92_is_active_finite_and_resource_closed():
    rows, labels = _support(5)
    state = fit_held_d92_head(
        rows, labels, old_classes=OLD, new_classes=NEW, k_shot=5
    )
    logits = score_held_d92_head(state, rows[:7])
    assert state.active is True
    assert logits.shape == (7, 6)
    assert np.isfinite(logits).all()
    receipt = d92_resource_receipt(state)
    assert receipt["full_head_state_bytes"] <= 262_144
    assert receipt["query_rows_used_for_fit"] == 0


def test_k1_is_exact_declared_qknn_fallback_not_a_fake_affine_head():
    rows, labels = _support(1)
    state = fit_held_d92_head(
        rows, labels, old_classes=OLD, new_classes=NEW, k_shot=1
    )
    assert state.active is False
    assert state.audit["status"] == "k1_exact_qknn_fallback"
    with pytest.raises(GRBJP4HeldHeadError, match="exact qKNN"):
        score_held_d92_head(state, rows)


def test_within_group_label_permutation_is_equivariant():
    rows, labels = _support(10)
    first = fit_held_d92_head(
        rows, labels, old_classes=OLD, new_classes=NEW, k_shot=10
    )
    mapping = dict(zip(OLD, reversed(OLD))) | {"n0": "n0"}
    renamed_labels = tuple(mapping[value] for value in labels)
    renamed_old = tuple(mapping[value] for value in OLD)
    second = fit_held_d92_head(
        rows,
        renamed_labels,
        old_classes=renamed_old,
        new_classes=("n0",),
        k_shot=10,
    )
    logits_a = score_held_d92_head(first, rows[:9])
    logits_b = score_held_d92_head(second, rows[:9])
    assert np.array_equal(logits_a, logits_b)


def test_unbalanced_or_overlapping_registry_fails_closed():
    rows, labels = _support(5)
    with pytest.raises(GRBJP4HeldHeadError, match="disjoint"):
        fit_held_d92_head(
            rows, labels, old_classes=OLD, new_classes=("o0",), k_shot=5
        )
    with pytest.raises(GRBJP4HeldHeadError, match="balanced"):
        fit_held_d92_head(
            rows[:-1], labels[:-1], old_classes=OLD, new_classes=NEW, k_shot=5
        )
