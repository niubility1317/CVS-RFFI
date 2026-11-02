import numpy as np
import pytest

from experiments.pamf_phase2_practical.milkr.r42_direct_c import fit_direct_c, source_prototypes, DirectPair
from experiments.pamf_phase2_practical.milkr import orbit_ensemble as orbit


def features(rows, seed):
    rng = np.random.default_rng(seed)
    dims = {'z_id': 160, 'fft': 96, 't_emb': 160, 'f_emb': 160, 'pa_local': 160, 'sig_views': 336}
    data = {name: rng.normal(size=(rows, dim)) for name, dim in dims.items()}
    data.update({'C4_' + name: rng.normal(size=(rows, dim)) for name, dim in dims.items()})
    p = rng.uniform(.1, 1., size=(rows, 6))
    data['teacher_probs'] = p / p.sum(axis=1, keepdims=True)
    return data


RULE = {'config': ('signature', 'all', 0., 'single'), 'ridge': .1}


def test_head_only_fits_new_support_and_scores_per_sample_without_batch_dependence():
    support = features(4, 7)
    state = fit_direct_c(support, [6, 6, 7, 7], ['a', 'b', 'c', 'd'], rule=RULE)
    query = features(200, 8)
    all_scores = state.score(query)
    one_scores = state.score({name: value[:1] for name, value in query.items()})
    assert all_scores.shape == (200, 8)
    np.testing.assert_allclose(all_scores[:1], one_scores, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(all_scores[:, :6].argmax(1), query['teacher_probs'].argmax(1))
    assert state.audit['actual_target_old_labels'] == 0
    assert state.audit['actual_target_new_labels'] == 4
    assert state.audit['query_fit'] is False


def test_source_prototypes_are_frozen_and_change_old_prior():
    support = features(4, 1)
    prototypes = features(6, 2)
    state = fit_direct_c(support, [6, 6, 7, 7], ['a', 'b', 'c', 'd'], rule=RULE, source_prototypes=prototypes)
    query = features(2, 3)
    score = state.score(query)
    p = orbit.normalize(prototypes)
    assert all(not value.flags.writeable for value in p.values())
    assert score.shape == (2, 8) and np.isfinite(score).all()
    assert state.audit['old_source'] == 'source_prototypes'


def test_rejects_old_target_support_and_duplicate_physical_ids():
    support = features(4, 5)
    with pytest.raises(ValueError, match='new class'):
        fit_direct_c(support, [0, 0, 6, 6], ['a', 'b', 'c', 'd'], rule=RULE)
    with pytest.raises(ValueError, match='support IDs'):
        fit_direct_c(support, [6, 6, 7, 7], ['a', 'a', 'c', 'd'], rule=RULE)


def test_source_prototypes_are_six_source_class_aggregates():
    source = features(12, 28)
    labels = np.repeat(np.arange(6), 2)
    prototypes = source_prototypes(source, labels)
    assert all(value.shape[0] == 6 for value in prototypes.values())
    expected = orbit.normalize(source)['z_id'][:2].mean(axis=0)
    np.testing.assert_allclose(prototypes['z_id'][0], expected)
    with pytest.raises(ValueError, match='source classes'):
        source_prototypes(source, np.zeros(12, dtype=np.int64))


def test_shared_kernel_scores_equal_both_independent_direct_states():
    support = features(10, 451)
    labels = np.repeat([6, 7], 5)
    ids = [str(i) for i in range(10)]
    head = fit_direct_c(support, labels, ids, rule=RULE)
    proto = fit_direct_c(support, labels, ids, rule=RULE, source_prototypes=features(6, 452))
    query = features(53, 453)
    actual_head, actual_proto = DirectPair(head, proto).score(query)
    np.testing.assert_allclose(actual_head, head.score(query), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual_proto, proto.score(query), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(actual_head.argmax(1), head.score(query).argmax(1))
    np.testing.assert_array_equal(actual_proto.argmax(1), proto.score(query).argmax(1))
