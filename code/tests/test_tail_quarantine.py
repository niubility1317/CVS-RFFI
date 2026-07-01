import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.tail_quarantine import (  # noqa: E402
    TailRegion,
    overflow_cap_loss,
    partition_tail_regions,
    region_to_ce_weights,
    tail_cvar_loss,
)


def test_tail_partition_and_weights_do_not_promote_extreme_tail():
    d = torch.tensor([1.0, 7.0, 13.0, 22.0])
    regions = partition_tail_regions(d, r_core_deg=5.0, r_accept_deg=10.0, r_tail_deg=15.0)
    weights = region_to_ce_weights(regions, soft_tail_weight=0.25, extreme_tail_weight=0.05)

    assert regions.tolist() == [
        TailRegion.CORE,
        TailRegion.SOFT_TAIL,
        TailRegion.EXTREME_TAIL,
        TailRegion.OUTSIDE,
    ]
    assert weights[0] == 1.0
    assert 0.0 < weights[1] < weights[0]
    assert 0.0 < weights[2] < weights[1]
    assert weights[3] == 0.0


def test_tail_losses_are_finite_and_penalize_overflow():
    d = torch.tensor([2.0, 5.0, 12.0, 18.0], requires_grad=True)
    loss = tail_cvar_loss(d, r_target_deg=10.0, top_frac=0.5) + overflow_cap_loss(d, r_accept_deg=10.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert d.grad is not None
    assert d.grad[-1] > d.grad[0]

