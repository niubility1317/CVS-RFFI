import numpy as np
import pytest
import torch

from experiments.pamf_phase2_practical.milkr.d92_branch_local_ridge import BranchLocalRidgeState


def test_worker_enables_cuda_only_when_requested(monkeypatch):
    from experiments.pamf_phase2_practical import r42_suite_worker as worker
    from experiments.pamf_phase2_practical.milkr import cuda_local_ridge

    state = object()
    marker = object()
    seen = []
    monkeypatch.setattr(cuda_local_ridge, 'CudaLocalRidge',
                        lambda value, device, **kw: seen.append((value, device, kw)) or marker)
    assert worker.local_ridge_scorer(state, 'cuda:0', False) is state
    assert worker.local_ridge_scorer(state, 'cuda:0', True) is marker
    assert worker.local_ridge_scorer(state, 'cuda:0', True, 16384, True) is marker
    assert seen == [(state, 'cuda:0', {'pair_budget': 2048, 'fast_transfer': False}),
                    (state, 'cuda:0', {'pair_budget': 16384, 'fast_transfer': True})]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for numerical comparison')
@pytest.mark.parametrize('tau', [0.0, 2.0])
@pytest.mark.parametrize('fast_transfer,pair_budget', [(False, 24), (True, 16384)])
def test_cuda_score_matches_original_including_old_class_ties(tau, fast_transfer, pair_budget):
    from experiments.pamf_phase2_practical.milkr.cuda_local_ridge import CudaLocalRidge

    rng = np.random.default_rng(711)
    n, classes, q = 12, 8, 7
    matrix = rng.normal(size=(n, classes))
    matrix[:, 1] = matrix[:, 0]
    state = BranchLocalRidgeState(
        rng.normal(size=(n, 256)), rng.normal(size=(n, 480)), matrix,
        rng.normal(size=n), 0.2, rng.normal(size=n), 0.3, tau, 1.0,
        tuple(f'C{i:02d}' for i in range(classes)), {},
    )
    features = {name: rng.normal(size=(q, size)) for name, size in (
        ('z_id', 160), ('fft', 96), ('t_emb', 160), ('f_emb', 160), ('pa_local', 160))}
    original = state.score(**features)
    accelerated = CudaLocalRidge(state, pair_budget=pair_budget, fast_transfer=fast_transfer)
    actual = accelerated.score(**features)
    np.testing.assert_allclose(actual, original, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(original, axis=1))
    np.testing.assert_array_equal(np.argmax(actual[:, :6], axis=1), np.argmax(original[:, :6], axis=1))
