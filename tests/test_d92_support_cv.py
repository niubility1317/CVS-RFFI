import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi.stage2_d92_support_cv import fit_support_cv, protect_old


def fixture(k=5, count=8):
    rng = np.random.default_rng(20260928)
    means = rng.normal(size=(count, 256))
    y = np.repeat(np.arange(count), k)
    x = means[y] + rng.normal(size=(len(y), 256)) * 0.4
    logits = x[:, :160] @ means[:6, :160].T / 10
    q = means + rng.normal(size=(count, 256)) * 0.4
    qlogits = q[:, :160] @ means[:6, :160].T / 10
    return x, y, logits, q, qlogits


def fit(x, y, logits, **kw):
    return fit_support_cv(support_features=x, support_labels=y, support_logits=logits,
        classes=[str(i) for i in range(len(set(y)))], fft_grid=(0.0, 1.0),
        ridge_grid=(1.0,), **kw)


@pytest.mark.parametrize('k', [1, 5, 10, 20])
def test_all_k_query_independent_and_support_unchanged(k):
    x, y, logits, q, qlogits = fixture(k)
    original = x.copy()
    state = fit(x, y, logits)
    snapshot = state.coefficient.copy()
    scores = state.score(q, qlogits)
    one = np.concatenate([state.score(q[i:i+1], qlogits[i:i+1]) for i in range(len(q))])
    np.testing.assert_allclose(scores, one, atol=1e-10)
    np.testing.assert_allclose(scores[::-1], state.score(q[::-1], qlogits[::-1]), atol=1e-10)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(state.coefficient, snapshot)
    assert state.audit['query_rows_used_for_fit'] == 0


def test_support_order_and_class_permutation():
    x, y, logits, q, qlogits = fixture()
    a = fit(x, y, logits)
    b = fit(x[::-1], y[::-1], logits[::-1])
    np.testing.assert_allclose(a.score(q, qlogits), b.score(q, qlogits), atol=1e-9)
    # Permute old/new registries independently, preserving known registration roles.
    perm = np.array([3, 1, 5, 0, 2, 4, 7, 6])
    inv = np.argsort(perm)
    c = fit(x, inv[y], logits[:, perm[:6]])
    np.testing.assert_allclose(a.score(q, qlogits)[:, perm], c.score(q, qlogits[:, perm[:6]]), atol=1e-8)


def test_k1_old_only_exact_frozen_dg_and_joint_mass_preserved():
    x, y, logits, q, qlogits = fixture(1, 6)
    state = fit(x, y, logits)
    np.testing.assert_array_equal(state.score(q, qlogits).argmax(1), qlogits.argmax(1))
    rng = np.random.default_rng(41)
    scores = rng.normal(size=(12, 26))
    frozen = rng.normal(size=(12, 6))
    before = np.exp(protect_old(scores, frozen, 6, 0))
    after = np.exp(protect_old(scores, frozen, 6, 0.75))
    np.testing.assert_allclose(before[:, :6].sum(1), after[:, :6].sum(1))
    np.testing.assert_allclose(before[:, 6:], after[:, 6:])
    np.testing.assert_allclose(after.sum(1), 1)


def test_singular_support_finite_and_wrong_contract_rejected():
    x, y, logits, q, qlogits = fixture()
    x[:] = 0
    state = fit(x, y, logits)
    assert np.isfinite(state.score(q, qlogits)).all()
    with pytest.raises(ValueError, match='Equal positive'):
        fit(x[:-1], y[:-1], logits[:-1])
    with pytest.raises(ValueError, match='labels/logits'):
        fit(x, y.astype(float), logits)
    with pytest.raises(TypeError):
        fit_support_cv(query_labels=np.zeros(8))


def test_fit_reads_only_support_even_when_query_changes():
    x, y, logits, q, qlogits = fixture()
    state = fit(x, y, logits)
    before = state.score(q, qlogits)
    state.score(q * 100, qlogits * -10)
    np.testing.assert_array_equal(before, state.score(q, qlogits))
    assert state.audit['loss'] is None
    assert all('harmonic' in t for t in state.audit['trace'])
