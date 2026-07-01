import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def test_soft_unknown_mixup_loss_uses_soft_labels_energy_and_vacuum_terms():
    from cvsrffi.losses import soft_unknown_mixup_loss

    z = torch.tensor(
        [
            [1.0, 0.00],
            [0.99, 0.05],
            [0.00, 1.0],
            [0.05, 0.99],
            [-1.0, 0.00],
            [-0.99, 0.05],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    logits = torch.tensor(
        [
            [5.0, -1.0, -1.0],
            [4.5, -0.5, -1.0],
            [-1.0, 5.0, -1.0],
            [-0.5, 4.5, -1.0],
            [-1.0, -1.0, 5.0],
            [-1.0, -0.5, 4.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss, info = soft_unknown_mixup_loss(
        z,
        y,
        logits=logits,
        mixup_count=4,
        mixup_order=3,
        alpha=0.5,
        ce_weight=1.0,
        energy_weight=1.0,
        energy_margin=0.5,
        vacuum_weight=1.0,
        vacuum_width_rad=math.radians(5.0),
        generator=torch.Generator().manual_seed(7),
    )

    assert loss.item() > 0.0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert info["soft_unknown_mixup_count"] == 4.0
    assert info["soft_unknown_mixup_order"] == 3.0
    assert info["soft_unknown_mixup_ce"] > 0.0
    assert info["soft_unknown_mixup_energy"] >= 0.0
    assert info["soft_unknown_mixup_vacuum"] >= 0.0
    assert 0.0 <= info["soft_unknown_mixup_virtual_accept_rate"] <= 1.0


def test_source_episode_three_sigma_loss_penalizes_mixup_overflow():
    from cvsrffi.losses import source_episode_three_sigma_loss, make_soft_unknown_mixup

    z = torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.02],
            [0.98, -0.02],
            [0.00, 1.00],
            [0.02, 0.99],
            [-0.02, 0.98],
            [-1.00, 0.00],
            [-0.99, 0.02],
            [-0.98, -0.02],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    d = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    mix = make_soft_unknown_mixup(
        z,
        y,
        count=4,
        alpha=0.5,
        mixup_order=3,
        generator=torch.Generator().manual_seed(3),
    )
    assert mix.source_labels.shape == (4, 3)
    for row in mix.source_labels:
        assert torch.unique(row).numel() == 3

    loss, info = source_episode_three_sigma_loss(
        z,
        y,
        d,
        radius_cap_rad=math.radians(6.0),
        mixup_features=mix.features,
        mixup_weight=1.0,
    )

    assert loss.item() > 0.0
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert info["source_episode_mixup_count"] == 4.0
    assert info["source_episode_mixup_order"] == 3.0
    assert info["source_episode_mixup_loss"] > 0.0
    assert info["source_episode_mixup_overflow_rate"] > 0.0
