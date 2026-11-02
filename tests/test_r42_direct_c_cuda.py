import numpy as np
import pytest
import torch

from experiments.pamf_phase2_practical.milkr.r42_direct_c import fit_direct_c
from experiments.pamf_phase2_practical.milkr.r42_direct_c_cuda import CudaDirectC
from experiments.pamf_phase2_practical.r42_direct_c_worker import cuda_for_support
from test_r42_direct_c_kernel import features

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


@pytest.mark.parametrize('geometry', ['signature', 'signature_identity', 'signature_historical'])
@pytest.mark.parametrize('old_source', ['head', 'prototype'])
@pytest.mark.parametrize('mix,scales', [(0., 'single'), (.5, 'multi')])
def test_cuda_matches_cpu_scores_and_class_decisions(geometry, old_source, mix, scales):
    support = features(10, 123)
    prototypes = features(6, 234) if old_source == 'prototype' else None
    rule = {'config': (geometry, 'all', mix, scales), 'ridge': .1}
    state = fit_direct_c(support, np.repeat([6, 7], 5), [str(i) for i in range(10)],
                         rule=rule, source_prototypes=prototypes)
    query = features(37, 345)
    expected = state.score(query)
    actual = CudaDirectC(state, 'cuda:0', pair_budget=100).score(query)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    np.testing.assert_array_equal(actual[:, :6].argmax(1), expected[:, :6].argmax(1))


def test_cuda_one_query_matches_scoring_in_a_batch():
    state = fit_direct_c(features(2, 11), [6, 7], ['a', 'b'],
        rule={'config': ('signature_identity', 'nearest', .5, 'single'), 'ridge': .01})
    query = features(11, 12)
    scorer = CudaDirectC(state, 'cuda:0', pair_budget=8)
    batch = scorer.score(query)
    single = scorer.score({k: v[:1] for k, v in query.items()})
    np.testing.assert_allclose(batch[:1], single, rtol=1e-11, atol=1e-11)


def test_execution_route_uses_only_support_size():
    rt = {'direct_c_cuda': True, 'direct_c_cuda_min_support': 256}
    assert not cuda_for_support(rt, 200)
    assert cuda_for_support(rt, 400)
    assert not cuda_for_support({}, 400)
