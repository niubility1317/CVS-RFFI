from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from torch.utils.data import Sampler


def _extract_item_fields(dataset, index: int, tx_key: str, domain_key: str) -> Tuple[int, int]:
    if hasattr(dataset, "index"):
        item = dataset.index[index]
        tx = getattr(item, tx_key, getattr(item, "tx_i", None))
        dom = getattr(item, domain_key, None)
        if dom is None and domain_key == "rx_day":
            rx = getattr(item, "rx_i", None)
            day = getattr(item, "day_i", 0)
            dom = int(rx) * 1000 + int(day) if rx is not None else None
        if dom is None:
            dom = getattr(item, "rx_i", None)
        if tx is not None and dom is not None:
            return int(tx), int(dom)
    sample = dataset[index]
    if isinstance(sample, Mapping):
        return int(sample[tx_key]), int(sample[domain_key])
    if isinstance(sample, (tuple, list)) and len(sample) >= 3:
        return int(sample[1]), int(sample[2])
    raise ValueError("dataset items must expose tx/domain metadata")


class BalancedTxDomainBatchSampler(Sampler[List[int]]):
    """Batch sampler producing N_TX x N_DOMAIN x N_PER_CELL batches."""

    def __init__(
        self,
        dataset,
        tx_per_batch: int,
        domain_per_batch: int,
        samples_per_tx_domain: int,
        *,
        replacement: bool = True,
        seed: int = 1337,
        tx_key: str = "tx_i",
        domain_key: str = "rx_day",
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.tx_per_batch = int(tx_per_batch)
        self.domain_per_batch = int(domain_per_batch)
        self.samples_per_tx_domain = int(samples_per_tx_domain)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.tx_key = str(tx_key)
        self.domain_key = str(domain_key)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        if self.tx_per_batch <= 0 or self.domain_per_batch <= 0 or self.samples_per_tx_domain <= 0:
            raise ValueError("tx_per_batch, domain_per_batch, and samples_per_tx_domain must be positive")
        self.cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.by_tx: Dict[int, List[int]] = defaultdict(list)
        self.by_domain: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            tx, domain = _extract_item_fields(dataset, idx, self.tx_key, self.domain_key)
            self.cells[(tx, domain)].append(idx)
            self.by_tx[tx].append(idx)
            self.by_domain[domain].append(idx)
        self.tx_values = sorted(self.by_tx)
        self.domain_values = sorted(self.by_domain)
        if not self.tx_values or not self.domain_values:
            raise ValueError("dataset must contain at least one tx and one domain")

    @property
    def batch_size(self) -> int:
        return self.tx_per_batch * self.domain_per_batch * self.samples_per_tx_domain

    def __len__(self) -> int:
        if self.replacement:
            return max(1, math.ceil(len(self.dataset) / float(max(1, self.batch_size))))
        return max(1, len(self.dataset) // max(1, self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch * 1000003)
        self._epoch += 1
        for _ in range(len(self)):
            txs = rng.sample(self.tx_values, k=min(self.tx_per_batch, len(self.tx_values)))
            domains = rng.sample(self.domain_values, k=min(self.domain_per_batch, len(self.domain_values)))
            batch: List[int] = []
            for tx in txs:
                for dom in domains:
                    pool = self.cells.get((tx, dom), [])
                    if not pool:
                        continue
                    if self.replacement or len(pool) < self.samples_per_tx_domain:
                        batch.extend(rng.choice(pool) for _ in range(self.samples_per_tx_domain))
                    else:
                        batch.extend(rng.sample(pool, k=self.samples_per_tx_domain))
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if batch:
                yield batch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def batch_geometry_stats(self, batch_indices: Sequence[int]) -> Dict[str, float]:
        txs = []
        domains = []
        replacement_count = len(batch_indices) - len(set(batch_indices))
        for idx in batch_indices:
            tx, domain = _extract_item_fields(self.dataset, int(idx), self.tx_key, self.domain_key)
            txs.append(tx)
            domains.append(domain)
        same_tx_cross_domain = 0
        same_domain_cross_tx = 0
        rectangles = 0
        for i in range(len(batch_indices)):
            for j in range(i + 1, len(batch_indices)):
                if txs[i] == txs[j] and domains[i] != domains[j]:
                    same_tx_cross_domain += 1
                if domains[i] == domains[j] and txs[i] != txs[j]:
                    same_domain_cross_tx += 1
        cells = {(tx, dom) for tx, dom in zip(txs, domains)}
        uniq_tx = sorted(set(txs))
        uniq_dom = sorted(set(domains))
        for a_i, tx1 in enumerate(uniq_tx):
            for tx2 in uniq_tx[a_i + 1 :]:
                for d_i, d1 in enumerate(uniq_dom):
                    for d2 in uniq_dom[d_i + 1 :]:
                        if {(tx1, d1), (tx1, d2), (tx2, d1), (tx2, d2)} <= cells:
                            rectangles += 1
        return {
            "tx_per_batch": float(len(set(txs))),
            "domain_per_batch": float(len(set(domains))),
            "same_tx_cross_domain_pairs": float(same_tx_cross_domain),
            "same_domain_cross_tx_pairs": float(same_domain_cross_tx),
            "tx_rx_rectangles": float(rectangles),
            "replacement_ratio": float(replacement_count) / float(max(1, len(batch_indices))),
        }
