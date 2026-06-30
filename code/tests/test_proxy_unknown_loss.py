import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import proxy_unknown_energy_loss  # noqa: E402


def test_proxy_unknown_energy_loss_keeps_holdout_auxiliary_and_penalizes_known_acceptance():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.05],
            [0.0, 1.0],
            [0.05, 0.98],
            [-1.0, 0.0],
            [-0.98, 0.05],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    loss, metrics = proxy_unknown_energy_loss(
        features,
        labels,
        holdout_label=2,
        virtual_count=4,
        energy_margin=0.25,
        placeholder_weight=0.5,
        virtual_detach=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["proxy_unknown_count"] == 2.0
    assert metrics["known_count"] == 4.0
    assert metrics["virtual_count"] == 4.0
    assert metrics["energy_proxy"] > metrics["energy_known"]
    assert 0.0 <= metrics["proxy_unknown_auc"] <= 1.0
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_proxy_unknown_energy_loss_is_graph_safe_when_no_holdout_samples():
    features = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])

    loss, metrics = proxy_unknown_energy_loss(features, labels, holdout_label=2)
    loss.backward()

    assert loss.item() == 0.0
    assert features.grad is not None
    assert metrics["active"] == 0.0
