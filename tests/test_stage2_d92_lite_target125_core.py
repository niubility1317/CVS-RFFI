from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi import stage2_d129_joint6_heads as d129
from cvsrffi import stage2_d92_lite_target125_core as core
from cvsrffi.stage2_diag_cosine_exploration import registered_feature
from cvsrffi.stage2_zid_student_t_qknn import score_zid_student_t_logits


OLD = tuple(f"old_{index}" for index in range(6))
NEW = tuple(f"new_{index}" for index in range(5))


def _features(rng: np.random.Generator, count: int) -> np.ndarray:
    zid = rng.normal(size=(count, 160)).astype(np.float32)
    zid /= np.linalg.norm(zid, axis=1, keepdims=True)
    aux = rng.normal(size=(count, 128)).astype(np.float32)
    aux /= np.linalg.norm(aux, axis=1, keepdims=True)
    return np.concatenate(
        [zid / np.sqrt(np.float32(17.0)), 4.0 * aux / np.sqrt(np.float32(17.0))],
        axis=1,
    ).astype(np.float32)


def _pair(k: int) -> core.D92LiteTarget125Pair:
    rng = np.random.default_rng(20260804 + k)
    return core.build_d92_lite_pair(
        _features(rng, len(OLD) * k),
        sum(([label] * k for label in OLD), []),
        OLD,
        _features(rng, len(NEW) * k),
        sum(([label] * k for label in NEW), []),
        NEW,
        seed=713102,
        device="cpu",
        d92_fit=lambda *_args, **_kwargs: None,
    )


def test_registered_feature_primary_recovers_canonical_zid_semantics() -> None:
    rng = np.random.default_rng(9)
    iq = rng.normal(size=(4, 2, 256)).astype(np.float32)
    zid = rng.normal(size=(4, 160)).astype(np.float32)
    actual = core.normalized_zid160_from_registered_feature(
        registered_feature(iq, zid)
    )
    expected = d129.normalize_zid160_rows(zid)
    assert np.allclose(actual, expected, rtol=0.0, atol=2.0e-6)
    assert not np.array_equal(actual, zid)


@pytest.mark.parametrize("k", [5, 10])
def test_before_is_qknn_and_after_is_lite_for_identifiable_k(k: int) -> None:
    pair = _pair(k)
    query = _features(np.random.default_rng(k), 3)
    assert pair.audit["before_head"] == "phase1_locked_student_t_qknn"
    assert pair.audit["after_head"] == d129.LITE_HEAD
    assert type(pair.after_lite_state) is d129.D129AffineHeadState
    assert core.score(pair, "before", core.TRANSPORT_ARM, query).shape == (3, 6)
    assert core.score(pair, "after", core.TRANSPORT_ARM, query).shape == (3, 11)


def test_k1_after_is_exact_qknn_logit_alias() -> None:
    pair = _pair(1)
    query288 = _features(np.random.default_rng(1), 4)
    query160 = core.normalized_zid160_from_registered_feature(query288)
    direct = score_zid_student_t_logits(
        pair.after_bank, query160, metric=pair.after_metric
    )
    actual = core.score(pair, "after", core.TRANSPORT_ARM, query288)
    assert pair.after_lite_state is None
    assert pair.audit["after_head"] == "exact_same_qknn_logits_alias"
    assert np.array_equal(actual, direct)


def test_public_score_has_no_truth_role_or_update_surface() -> None:
    forbidden = {"truth", "query_truth", "role", "quota", "query_labels", "update"}
    assert not forbidden.intersection(inspect.signature(core.score).parameters)
    pair = _pair(5)
    query = _features(np.random.default_rng(2), 2)
    with pytest.raises(core.D92LiteTarget125CoreError):
        core.score(pair, "after", "M_HEAD", query)
    with pytest.raises(core.D92LiteTarget125CoreError):
        core.score(pair, "unknown", core.TRANSPORT_ARM, query)


def _lite_logits(*rows: tuple[float, ...]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_non_tied_lite_logits_are_bitwise_unchanged_and_skip_qknn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair(5)
    query = np.zeros((2, 160), dtype=np.float32)
    logits = np.tile(np.arange(11, dtype=np.float32), (2, 1))

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("qKNN must not run without an exact Lite top tie")

    monkeypatch.setattr(core, "score_zid_student_t_logits", forbidden)
    actual = core._resolve_exact_lite_top_ties(pair, query, logits)
    assert np.array_equal(actual, logits)


def test_lite_top_tie_uses_only_tied_qknn_winner_and_changes_one_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair(5)
    query = np.zeros((1, 160), dtype=np.float32)
    logits = _lite_logits((4.0, 4.0, 3.5, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0))
    logits[0, 3:] = np.float32(0.0)
    qknn = _lite_logits((1.0, 2.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(
        core,
        "score_zid_student_t_logits",
        lambda *_args, **_kwargs: qknn,
    )
    actual = core._resolve_exact_lite_top_ties(pair, query, logits)
    changed = np.flatnonzero(actual[0] != logits[0])
    assert changed.tolist() == [1]
    assert actual[0, 1] == np.nextafter(np.float32(4.0), np.float32(np.inf))
    assert np.sum(actual == np.max(actual, axis=1, keepdims=True), axis=1).tolist() == [1]


def test_lite_and_qknn_double_top_tie_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair(5)
    query = np.zeros((1, 160), dtype=np.float32)
    logits = np.zeros((1, 11), dtype=np.float32)
    logits[0, :2] = np.float32(1.0)
    qknn = np.zeros((1, 11), dtype=np.float32)
    qknn[0, :2] = np.float32(2.0)
    monkeypatch.setattr(
        core,
        "score_zid_student_t_logits",
        lambda *_args, **_kwargs: qknn,
    )
    with pytest.raises(core.D92LiteTarget125CoreError, match="remains tied"):
        core._resolve_exact_lite_top_ties(pair, query, logits)


def test_exact_top_tie_resolution_is_class_permutation_equivariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair(5)
    query = np.zeros((1, 160), dtype=np.float32)
    logits = np.zeros((1, 11), dtype=np.float32)
    logits[0, [2, 7]] = np.float32(3.0)
    qknn = np.arange(11, dtype=np.float32)[None, :]
    monkeypatch.setattr(
        core,
        "score_zid_student_t_logits",
        lambda *_args, **_kwargs: qknn,
    )
    reference = core._resolve_exact_lite_top_ties(pair, query, logits)

    permutation = np.asarray([7, 2, 10, 0, 1, 3, 4, 5, 6, 8, 9], dtype=np.int64)
    monkeypatch.setattr(
        core,
        "score_zid_student_t_logits",
        lambda *_args, **_kwargs: qknn[:, permutation],
    )
    permuted = core._resolve_exact_lite_top_ties(
        pair, query, logits[:, permutation]
    )
    assert np.array_equal(permuted, reference[:, permutation])


def test_exact_top_tie_resolution_is_query_batch_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _pair(5)
    query = np.zeros((2, 160), dtype=np.float32)
    query[:, 0] = np.asarray([1.0, -1.0], dtype=np.float32)
    logits = np.zeros((2, 11), dtype=np.float32)
    logits[:, :2] = np.float32(5.0)

    def qknn_from_query(_bank: object, rows: np.ndarray, **_kwargs: object) -> np.ndarray:
        values = np.zeros((len(rows), 11), dtype=np.float32)
        values[:, 0] = rows[:, 0]
        values[:, 1] = -rows[:, 0]
        return values

    monkeypatch.setattr(core, "score_zid_student_t_logits", qknn_from_query)
    reference = core._resolve_exact_lite_top_ties(pair, query, logits)
    order = np.asarray([1, 0], dtype=np.int64)
    permuted = core._resolve_exact_lite_top_ties(pair, query[order], logits[order])
    assert np.array_equal(permuted, reference[order])


def test_before_is_exact_qknn_logit_path() -> None:
    pair = _pair(5)
    query288 = _features(np.random.default_rng(44), 3)
    query160 = core.normalized_zid160_from_registered_feature(query288)
    direct = score_zid_student_t_logits(
        pair.before_bank, query160, metric=pair.before_metric
    )
    actual = core.score(pair, "before", core.TRANSPORT_ARM, query288)
    assert np.array_equal(actual, direct)
