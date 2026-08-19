import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import _labeled_channel_pair_invariance_loss  # noqa: E402


def test_labeled_channel_pair_invariance_aligns_clean_and_satellite_views():
    clean_z_id = torch.tensor(
        [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]],
        dtype=torch.float32,
        requires_grad=True,
    )
    sat_z_id = torch.tensor(
        [[0.94, 0.34], [0.96, 0.28], [0.34, 0.94], [0.28, 0.96]],
        dtype=torch.float32,
        requires_grad=True,
    )
    tx_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss, metrics = _labeled_channel_pair_invariance_loss(
        clean_z_id,
        sat_z_id,
        tx_labels,
        channel_weight=1.0,
        channel_pair_weight=1.0,
        min_groups=2,
        min_samples_per_group=1,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert metrics["active"] == 1.0
    assert metrics["channel_active"] == 1.0
    assert metrics["channel_pair_active"] == 1.0
    assert metrics["receiver_active"] == 0.0
    assert metrics["day_active"] == 0.0
    assert metrics["channel_pair_count"] == 4.0
    assert metrics["channel_pair_loss"] > 0.0

    loss.backward()
    assert clean_z_id.grad is not None
    assert sat_z_id.grad is not None
    assert torch.isfinite(clean_z_id.grad).all()
    assert torch.isfinite(sat_z_id.grad).all()
