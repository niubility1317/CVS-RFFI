import torch

from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior
from experiments.phase1_dura_orthogonal.source_masked_representation_probe import MASK_START
from experiments.phase1_dura_orthogonal.source_masked_temporal_probe import descriptors


def test_temporal_descriptors_ignore_masked_samples():
    torch.manual_seed(8)
    model = MaskedIQPrior().eval()
    x = torch.randn(3, 2, 256)
    changed = x.clone()
    changed[:, :, MASK_START:MASK_START+32] += 1000
    first, second = descriptors(model, x), descriptors(model, changed)
    assert set(first) == {'global', 'temporal_8', 'temporal_32', 'temporal_256'}
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
