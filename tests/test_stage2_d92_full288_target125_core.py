from __future__ import annotations

import numpy as np
import pytest

from cvsrffi import stage2_d92_full288_target125_core as core


def _unit_rows(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(count, core.FEATURE_WIDTH)).astype(np.float32)
    rows /= np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True)
    return rows


def _support(k: int) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray, tuple[str, ...], tuple[str, ...]]:
    old = tuple(f"old{i}" for i in range(6))
    new = tuple(f"new{i}" for i in range(5))
    old_rows = _unit_rows(11 + k, len(old) * k)
    new_rows = _unit_rows(41 + k, len(new) * k)
    old_labels = tuple(label for label in old for _ in range(k))
    new_labels = tuple(label for label in new for _ in range(k))
    return old_rows, old_labels, old, new_rows, new_labels, new


@pytest.mark.parametrize("k", [1, 5, 10])
def test_full288_pair_scores_all_registered_classes(k: int) -> None:
    old_rows, old_labels, old, new_rows, new_labels, new = _support(k)
    pair = core.build_d92_full288_pair(
        old_rows,
        old_labels,
        old,
        new_rows,
        new_labels,
        new,
        seed=713102,
        device="cpu",
        d92_fit=None,
    )
    query = _unit_rows(100 + k, 4)
    before = core.score(pair, "before", core.TRANSPORT_ARM, query)
    after = core.score(pair, "after", core.TRANSPORT_ARM, query)
    assert before.shape == (4, 6)
    assert after.shape == (4, 11)
    assert np.isfinite(before).all()
    assert np.isfinite(after).all()


def test_full288_tie_uses_support_fingerprint_not_registry_order() -> None:
    raw = np.asarray([[1.0, 1.0]], dtype=np.float64)
    secondary = np.asarray([[0.5, 0.5]], dtype=np.float64)
    result = core._resolve_ties(raw, secondary, ((0.0,), (1.0,)))
    assert result[0, 1] > result[0, 0]


def test_identical_support_fingerprint_still_fails_closed() -> None:
    raw = np.asarray([[1.0, 1.0]], dtype=np.float64)
    secondary = np.asarray([[0.5, 0.5]], dtype=np.float64)
    with pytest.raises(core.D92Full288CoreError, match="identical full-288"):
        core._resolve_ties(raw, secondary, ((1.0,), (1.0,)))
