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
