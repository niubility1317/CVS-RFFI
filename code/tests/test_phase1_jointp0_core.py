from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.balanced_tx_rx_sampler import BalancedTxDomainBatchSampler  # noqa: E402
from cvsrffi.losses import PrototypeMemoryBank  # noqa: E402
from post_stage_cli import add_sat_eval_args  # noqa: E402


@dataclass(frozen=True)
class _Item:
    tx_i: int
    rx_i: int
    day_i: int


class _GridDataset:
    def __init__(self, tx_count: int = 6, domain_count: int = 6, per_cell: int = 4):
        self.index = [
            _Item(tx_i=tx, rx_i=domain, day_i=0)
            for tx in range(tx_count)
            for domain in range(domain_count)
            for _ in range(per_cell)
        ]

    def __len__(self):
        return len(self.index)


def test_balanced_sampler_produces_complete_tx_domain_grid_and_changes_by_epoch():
    sampler = BalancedTxDomainBatchSampler(
        _GridDataset(),
        tx_per_batch=6,
        domain_per_batch=6,
        samples_per_tx_domain=3,
        seed=17,
        drop_last=True,
    )
    sampler.set_epoch(1)
    first = next(iter(sampler))
    sampler.set_epoch(2)
    second = next(iter(sampler))
    assert len(first) == 108
    assert len(second) == 108
    assert first != second
    stats = sampler.batch_geometry_stats(first)
    assert stats["tx_per_batch"] == 6.0
    assert stats["domain_per_batch"] == 6.0
    assert stats["tx_rx_rectangles"] > 0.0


def test_prototype_domain_alignment_is_differentiable_on_current_features():
    bank = PrototypeMemoryBank(
        2,
        2,
        momentum=0.9,
        margin=0.1,
        domain_align_weight=1.0,
        push_weight=0.0,
        min_count=2,
    )
    init_z = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    )
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    d = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    bank.update(init_z, y, d)
    z = init_z.clone().requires_grad_(True)
    loss, info = bank.loss(z, y, d)
    loss.backward()
    assert info["proto_domain_align"] > 0.0
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.norm().item()) > 0.0


def test_default_satellite_evaluation_protocol_is_leo_weak():
    parser = argparse.ArgumentParser()
    add_sat_eval_args(parser)
    args = parser.parse_args([])
    assert args.eval_sat_scenarios == "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
