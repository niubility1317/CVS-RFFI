from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class DomainEpisode:
    meta_train_indices: List[int]
    meta_val_indices: List[int]
    meta_train_domains: List[int]
    meta_val_domain: int


def sample_rxday_episode(
    domain_ids: Sequence[int],
    *,
    seed: int = 0,
    meta_train_domain_count: int = 2,
    max_samples_per_domain: int = 0,
) -> DomainEpisode:
    """Sample a source-domain episode for first-order MLDG style checks."""
    by_domain: Dict[int, List[int]] = {}
    for idx, did in enumerate(domain_ids):
        by_domain.setdefault(int(did), []).append(int(idx))
    domains = sorted(by_domain)
    if len(domains) < 2:
        raise ValueError("Meta-SSL MLDG episodes require at least two source domains.")

    rng = np.random.default_rng(int(seed))
    shuffled = [domains[int(i)] for i in rng.permutation(len(domains)).tolist()]
    val_domain = int(shuffled[0])
    train_domains = [int(d) for d in shuffled[1:1 + max(1, int(meta_train_domain_count))]]
    if not train_domains:
        train_domains = [int(shuffled[1])]

    def pick(domain: int) -> List[int]:
        values = list(by_domain[int(domain)])
        if int(max_samples_per_domain) > 0 and len(values) > int(max_samples_per_domain):
            take = rng.permutation(len(values))[: int(max_samples_per_domain)].tolist()
            values = sorted(values[int(i)] for i in take)
        return values

    meta_train: List[int] = []
    for domain in train_domains:
        meta_train.extend(pick(domain))
    meta_val = pick(val_domain)
    if not meta_train or not meta_val:
        raise ValueError("Sampled Meta-SSL episode is empty.")
    return DomainEpisode(
        meta_train_indices=sorted(meta_train),
        meta_val_indices=sorted(meta_val),
        meta_train_domains=sorted(train_domains),
        meta_val_domain=val_domain,
    )
