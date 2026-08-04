from __future__ import annotations

import numpy as np
import pytest

from cvsrffi import stage2_d92_pr160_core as core
from cvsrffi.stage2_d92_pr160_runtime import normalize_signed_prerelu160


def _unit_rows(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(count, 160)).astype(np.float32)
    rows /= np.sqrt(np.sum(rows.astype(np.float64) ** 2, axis=1))[:, None]
    return rows.astype(np.float32)


def _support(k: int = 5) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    classes = tuple(f"old-{i}" for i in range(6)) + tuple(f"new-{i}" for i in range(5))
    rows = _unit_rows(1801, len(classes) * k)
    labels = tuple(label for label in classes for _ in range(k))
    return rows, labels, classes


def test_signed_totalization_uses_positive_view_then_signed_fallback() -> None:
    positive = np.zeros((2, 160), dtype=np.float32)
    positive[0, 0] = 2.0
    positive[0, 1] = -1.0
    positive[1, :] = -1.0
    result = normalize_signed_prerelu160(positive)
    assert np.isclose(result[0, 0], 1.0)
    assert np.all(result[0, 1:] == 0.0)
    assert np.allclose(result[1], np.full(160, -1.0 / np.sqrt(160.0), dtype=np.float32))


@pytest.mark.parametrize("bad", [np.zeros((1, 160), dtype=np.float32), np.full((1, 160), np.nan, dtype=np.float32)])
def test_signed_totalization_rejects_zero_or_nonfinite(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="zero|non-finite|finite"):
        normalize_signed_prerelu160(bad)


def test_k1_is_exact_qknn_alias_and_k5_uses_one_all_class_head() -> None:
    rows, labels, classes = _support(5)
    old_rows, old_labels = rows[: 6 * 5], labels[: 6 * 5]
    new_rows, new_labels = rows[6 * 5 :], labels[6 * 5 :]
    pair = core.build_d92_lite_pair(
        old_rows,
        old_labels,
        classes[:6],
        new_rows,
        new_labels,
        classes[6:],
        seed=713102,
        device="cpu",
        d92_fit=None,
    )
    assert pair.active_k == 5
    assert pair.after_lite_state is not None
    assert pair.after_lite_state.fit_receipt["old_new_role_access"] is False
    query = _unit_rows(1802, 7)
    logits = core.score(pair, "after", core.TRANSPORT_ARM, query)
    assert logits.shape == (7, len(classes))
    assert np.isfinite(logits).all()

    one_rows, one_labels, one_classes = _support(1)
    one_pair = core.build_d92_lite_pair(
        one_rows[:6],
        one_labels[:6],
        one_classes[:6],
        one_rows[6:],
        one_labels[6:],
        one_classes[6:],
        seed=713102,
        device="cpu",
        d92_fit=None,
    )
    assert one_pair.after_lite_state is None
    assert core.score(one_pair, "after", core.TRANSPORT_ARM, _unit_rows(1803, 3)).shape == (3, 11)


def test_exact_float32_tie_fails_closed_without_a_tie_key() -> None:
    with pytest.raises(core.D92PR160CoreError, match="TIE_UNRESOLVED"):
        core._require_unique_top(np.asarray([[1.0, 1.0]], dtype=np.float32))


def test_float32_precision_alias_tie_uses_unique_precast_winner() -> None:
    raw = np.asarray([[1.0 + 1.0e-8, 1.0 + 2.0e-8]], dtype=np.float64)
    rounded = raw.astype(np.float32)
    assert np.array_equal(rounded, np.asarray([[1.0, 1.0]], dtype=np.float32))
    resolved = core._resolve_float32_precision_alias_ties(raw, rounded)
    assert int(np.argmax(resolved[0])) == 1
    assert resolved[0, 1] > resolved[0, 0]


def test_float32_precision_alias_tie_still_fails_on_raw_tie() -> None:
    raw = np.asarray([[1.0, 1.0]], dtype=np.float64)
    with pytest.raises(core.D92PR160CoreError, match="remains tied"):
        core._resolve_float32_precision_alias_ties(raw, raw.astype(np.float32))


def test_raw_support_centroid_is_the_only_secondary_for_a_true_score_tie() -> None:
    raw = np.asarray([[1.0, 1.0]], dtype=np.float64)
    rounded = raw.astype(np.float32)
    secondary = np.asarray([[0.25, 0.5]], dtype=np.float64)
    resolved = core._resolve_float32_precision_alias_ties(raw, rounded, secondary)
    assert int(np.argmax(resolved[0])) == 1
    assert resolved[0, 1] > resolved[0, 0]


def test_raw_support_centroid_tie_remains_fail_closed() -> None:
    raw = np.asarray([[1.0, 1.0]], dtype=np.float64)
    secondary = np.asarray([[0.5, 0.5]], dtype=np.float64)
    with pytest.raises(core.D92PR160CoreError, match="raw support centroid"):
        core._resolve_float32_precision_alias_ties(raw, raw.astype(np.float32), secondary)
