import torch
import numpy as np
import pytest

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.dura_real_gain_r17 import R17Gain
from experiments.pamf_lora_main.dura_real_gain_r17_screen import select_unexposed


def test_r17_fusion_zero_initialization_and_two_trainable_directions():
    torch.manual_seed(51)
    y = torch.randn(2, 256, dtype=torch.complex64)
    prior = .7 * y
    r17 = -.2 * y
    action = Action(torch.ones(2, 1, dtype=y.dtype), torch.ones_like(y), torch.zeros_like(y))
    model = R17Gain(32)
    restored = model(y, prior, r17, action)
    torch.testing.assert_close(restored, prior, rtol=0, atol=0)
    restored.abs().square().mean().backward()
    gradients = model.net[-1].weight.grad.flatten(1).norm(dim=1)
    assert (gradients > 0).all() and torch.isfinite(gradients).all()


def test_screen_selection_excludes_previous_exposed_ids():
    rx = np.r_[np.zeros(400, dtype=np.int64), np.ones(50, dtype=np.int64)]
    prior = np.arange(60)
    first = select_unexposed(rx, prior)
    assert len(first) == 300 and not np.intersect1d(first, prior).size
    assert (rx[first] == 0).all()
    np.testing.assert_array_equal(first, select_unexposed(rx, prior))
    with pytest.raises(ValueError):
        select_unexposed(rx[:200], prior)
