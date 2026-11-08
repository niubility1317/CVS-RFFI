import pytest
import torch

from experiments.pamf_phase3_restoration.feature_response_reuse_probe import (
    cache_source_features, compare_one_epoch,
)
from experiments.pamf_phase3_restoration.response_joint_prior import selected_perturbation


class Reader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(512, 160)

    def features(self, x):
        return self.projection(x.flatten(1))


@pytest.mark.parametrize('u_index', [6300, 6301])
def test_four_direction_cache_matches_online_features_and_gradients(u_index):
    torch.manual_seed(17)
    reader = Reader().eval().requires_grad_(False)
    clean = torch.randn(2, 2, 256)
    received = torch.randn(2, 2, 256)
    indices = torch.tensor([0, u_index])
    scale = torch.ones(160)
    with torch.no_grad():
        target = reader.features(clean)
        cache = cache_source_features(reader, clean, indices)
        for epoch in range(4):
            delta, direction = selected_perturbation(clean, indices, epoch)
            expected = reader.features(clean + delta)
            torch.testing.assert_close(cache[torch.arange(2), direction], expected, rtol=0, atol=0)
    for epoch in range(4):
        result = compare_one_epoch(reader, clean, received, indices, scale, target, cache, epoch)
        assert result['loss_max_abs'] < 1e-6
        assert result['parameter_gradient_max_abs'] < 1e-5
        assert result['updated_parameter_max_abs'] < 1e-6
    assert all(p.grad is None for p in reader.parameters())


def test_training_reader_is_rejected():
    reader = Reader().train().requires_grad_(False)
    clean = torch.randn(2, 2, 256)
    with pytest.raises(ValueError, match='frozen evaluation'):
        cache_source_features(reader, clean, torch.tensor([0, 6300]))
