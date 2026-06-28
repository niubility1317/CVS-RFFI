import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.balanced_tx_rx_sampler import BalancedTxDomainBatchSampler  # noqa: E402


@dataclass
class Item:
    tx_i: int
    rx_i: int
    day_i: int = 0


class TinyDataset:
    def __init__(self):
        self.index = [Item(tx, rx, 0) for tx in range(3) for rx in range(2) for _ in range(2)]

    def __len__(self):
        return len(self.index)


def test_balanced_tx_domain_sampler_reports_batch_geometry():
    ds = TinyDataset()
    sampler = BalancedTxDomainBatchSampler(ds, tx_per_batch=2, domain_per_batch=2, samples_per_tx_domain=1, seed=7)
    batch = next(iter(sampler))
    stats = sampler.batch_geometry_stats(batch)

    assert len(batch) == 4
    assert stats["tx_per_batch"] == 2.0
    assert stats["domain_per_batch"] == 2.0
    assert stats["tx_rx_rectangles"] >= 1.0
    assert stats["replacement_ratio"] == 0.0
