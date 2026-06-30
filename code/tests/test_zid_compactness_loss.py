import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.losses import zid_compactness_loss  # noqa: E402


def test_zid_compactness_loss_penalizes_tail_cvar_and_reports_quantiles():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    compact = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.985, 0.174],
            [0.966, 0.259],
            [0.0, 1.0],
            [0.087, 0.996],
            [0.174, 0.985],
            [0.259, 0.966],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    tailed = torch.tensor(
        [
            [1.0, 0.0],
            [0.996, 0.087],
            [0.985, 0.174],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.087, 0.996],
            [0.174, 0.985],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    compact_loss, compact_metrics = zid_compactness_loss(
        compact,
        labels,
        domains,
        radius_rad=math.radians(35.0),
        cvar_alpha=0.75,
        supcon_weight=0.1,
        radius_weight=0.4,
        cvar_weight=0.5,
    )
    tail_loss, tail_metrics = zid_compactness_loss(
        tailed,
        labels,
        domains,
        radius_rad=math.radians(35.0),
        cvar_alpha=0.75,
        supcon_weight=0.1,
        radius_weight=0.4,
        cvar_weight=0.5,
    )
    tail_loss.backward()

    assert tail_loss.item() > compact_loss.item()
    assert tail_metrics["tail_cvar_deg"] > compact_metrics["tail_cvar_deg"]
    assert tail_metrics["pos_angle_p99_deg"] >= tail_metrics["pos_angle_p95_deg"]
    assert tail_metrics["active_classes"] == 2.0
    assert tailed.grad is not None
    assert torch.isfinite(tailed.grad).all()


def test_zid_compactness_loss_returns_graph_safe_zero_without_enough_classes():
    features = torch.randn(3, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 0])

    loss, metrics = zid_compactness_loss(features, labels, min_classes=2)
    loss.backward()

    assert loss.item() == 0.0
    assert features.grad is not None
    assert metrics["active_classes"] == 1.0
