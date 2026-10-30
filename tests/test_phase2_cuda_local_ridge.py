from dataclasses import replace
import numpy as np
import pytest
import torch

from experiments.pamf_phase2_practical.milkr.cuda_local_ridge import CudaLocalRidge
from experiments.pamf_phase2_practical.milkr.d92_branch_local_ridge import fit_branch_local_ridge

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def features(rng, n):
    return {key: rng.normal(size=(n, dim)).astype('float32') for key, dim in (
        ('z_id', 160), ('fft', 96), ('t_emb', 160), ('f_emb', 160), ('pa_local', 160))}


@pytest.mark.parametrize('classes,k', [(6, 1), (6, 5), (26, 5), (26, 20)])
def test_cuda_matches_cpu_and_is_independent_of_query_batch(classes, k):
    rng = np.random.default_rng(5432+classes+k)
    support = features(rng, classes*k)
    names = tuple(f'class_{i:02d}' for i in range(classes))
    state = fit_branch_local_ridge(**support, support_labels=np.repeat(np.arange(classes), k),
        support_ids=[f'physical_{i:04d}' for i in range(classes*k)], classes=names)
    query = features(rng, 19)
    query = {name: np.concatenate((value, support[name][:2])) for name, value in query.items()}
    cpu = state.score(**query)
    gpu = CudaLocalRidge(state, pair_budget=256)
    out = gpu.score(**query)
    np.testing.assert_allclose(out, cpu, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(out.argmax(1), cpu.argmax(1))
    np.testing.assert_array_equal(out[:, :6].argmax(1), cpu[:, :6].argmax(1))
    single = np.concatenate([gpu.score(**{name: v[i:i+1] for name, v in query.items()})
                             for i in range(len(cpu))])
    np.testing.assert_allclose(single, out, rtol=1e-10, atol=1e-10)


def test_identical_support_and_query_has_exact_tie():
    rng = np.random.default_rng(112)
    one = features(rng, 1)
    support = {k: np.repeat(v, 6, axis=0) for k, v in one.items()}
    state = fit_branch_local_ridge(**support, support_labels=np.arange(6),
        support_ids=[f'p{i}' for i in range(6)], classes=[f'c{i}' for i in range(6)])
    cpu = state.score(**one)
    gpu = CudaLocalRidge(state).score(**one)
    np.testing.assert_array_equal(gpu, cpu)
    assert gpu.argmax(1).item() == 0


def test_zero_bandwidth_and_close_scores_preserve_cpu_decisions():
    rng = np.random.default_rng(113)
    support = features(rng, 6)
    for value in support.values():
        value[:4] = value[0]
    state = fit_branch_local_ridge(**support, support_labels=np.arange(6),
        support_ids=[f'p{i}' for i in range(6)], classes=[f'c{i}' for i in range(6)])
    assert state.bandwidth_tau == 0
    query = features(rng, 3)
    query = {name: np.concatenate((value, support[name][:1])) for name, value in query.items()}
    gpu = CudaLocalRidge(state)
    np.testing.assert_allclose(gpu.score(**query), state.score(**query), atol=1e-12, rtol=1e-12)
    tied = replace(state, alpha=np.zeros_like(state.alpha))
    scorer = CudaLocalRidge(tied)
    np.testing.assert_array_equal(scorer.score(**query), tied.score(**query))
    assert scorer.cpu_tie_rows == len(next(iter(query.values())))
