from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.cric_dura_comparison import evaluation
from experiments.pamf_wisig_branch_style import transforms


def paired(transform, feature, module=None):
    feature.requires_grad_()
    valid = torch.tensor([True, False, True])
    result = transform(module, feature, torch.tensor([1, 1, 0]),
                       torch.full((3, 1, 1), .8), valid)
    assert torch.equal(result[1], feature[1])
    assert not torch.equal(result[0], feature[0])
    result.square().mean().backward()
    assert torch.isfinite(result).all() and torch.isfinite(feature.grad).all()
    return result.detach(), feature.detach()


def test_frequency_envelope_changes_only_low_order_band_modes():
    torch.manual_seed(24)
    result, original = paired(transforms.frequency_envelope, torch.randn(3, 8, 16))
    shift = (result - original).double()
    position = (torch.arange(16, dtype=torch.double) + .5) / 16
    high_mode = (3 * torch.pi * position).cos()
    assert torch.allclose(shift @ high_mode, torch.zeros(3, 8, dtype=torch.double), atol=1e-6)
    assert torch.all(shift.square().mean(-1).sqrt() <= .25 * original.square().mean(-1).sqrt())


def test_pa_gain_respects_every_order_and_memory_lag():
    torch.manual_seed(24)
    result, original = paired(transforms.pa_envelope, torch.randn(3, 12, 256),
                              SimpleNamespace(memory_depth=2, orders=(1, 3, 5)))
    gain = (result / original).reshape(3, 2, 6, 256)
    assert torch.allclose(gain, gain[:, :, :1].expand_as(gain), atol=2e-7)
    assert torch.allclose(gain[:, 0, :, :-1], gain[:, 1, :, 1:], atol=2e-7)
    assert gain.min() >= torch.exp(torch.tensor(-.22))
    assert gain.max() <= torch.exp(torch.tensor(.22))


def test_reference_auxiliary_coordinates_and_signature_endpoints_unchanged():
    torch.manual_seed(24)
    reference = torch.randn(3, 29) / 3
    reference[:, 27] = 0
    result, original = paired(transforms.reference_trend, reference)
    assert torch.equal(result[:, 24:], original[:, 24:])
    shift = (result[:, :24] - original[:, :24]).reshape(3, 2, 12).norm(dim=1)
    assert shift.max() <= .15 + 1e-6
    result, original = paired(transforms.signature_bend, torch.randn(3, 6, 32))
    boundaries = [offset + position for offset in (0, 8, 16, 24) for position in (0, 7)]
    assert torch.equal(result[..., boundaries], original[..., boundaries])


def test_reference_id_mismatch_blocks_all_truth_io():
    with patch.object(evaluation, 'preflight', side_effect=[
        {'cross_rx': (np.array(['opaque-a']), {})}, {'cross_rx': (np.array(['opaque-b']), {})}
    ]), patch.object(evaluation, 'read') as read:
        with pytest.raises(ValueError, match='paired reference coverage'):
            evaluation.score({'classes': ['a', 'b']}, reference={'classes': ['a', 'b']})
        read.assert_not_called()
