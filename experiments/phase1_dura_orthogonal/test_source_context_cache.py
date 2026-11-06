import pytest
import torch

from experiments.phase1_dura_orthogonal.source_context_cache import save_cache, load_cache


def test_cache_exact_and_rejects_mismatched_source(tmp_path):
    fit = [10, 20]
    data = (torch.randn(6, 2, 256), torch.randn(6, 2, 256),
            torch.randn(6, 2, 256), torch.ones(6), torch.randn(6, 2, 256))
    path = tmp_path/'context.pt'
    save_cache(path, data, {'clean': 2, 'practical': 4}, fit, 32, 'cuda:0')
    restored, counts = load_cache(path, fit, 32, 'cuda:0')
    assert all(torch.equal(x, y) for x, y in zip(data, restored))
    assert counts == {'clean': 2, 'practical': 4}
    with pytest.raises(ValueError):
        load_cache(path, [10, 21], 32, 'cuda:0')
    with pytest.raises(ValueError):
        load_cache(path, fit, 64, 'cuda:0')
    with pytest.raises(ValueError):
        load_cache(path, fit, 32, 'cuda:0', deterministic=True)
    with pytest.raises(FileExistsError):
        save_cache(path, data, counts, fit, 32, 'cuda:0')
    deterministic_path = tmp_path/'deterministic.pt'
    save_cache(deterministic_path, data, counts, fit, 32, 'cuda:0', deterministic=True)
    deterministic_data, _ = load_cache(deterministic_path, fit, 32, 'cuda:0', deterministic=True)
    assert all(torch.equal(x, y) for x, y in zip(data, deterministic_data))
    with pytest.raises(ValueError):
        load_cache(deterministic_path, fit, 32, 'cuda:0')
