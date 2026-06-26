from __future__ import annotations

from collections import OrderedDict
from collections import defaultdict
from typing import Iterable, Mapping, Set

import torch


def _aggregation_weights(client_ids, client_num_samples: Mapping[str, int], agg_weight: str):
    if str(agg_weight) == "uniform":
        return {cid: 1.0 / len(client_ids) for cid in client_ids}
    if str(agg_weight) != "num_samples":
        raise ValueError("agg_weight must be one of: num_samples, uniform")
    total = float(sum(max(0, int(client_num_samples[cid])) for cid in client_ids))
    if total <= 0:
        raise ValueError("client_num_samples must sum to a positive value")
    return {cid: max(0, int(client_num_samples[cid])) / total for cid in client_ids}


def domain_balanced_aggregation_weights(client_ids, domain_ids: Mapping[str, str]):
    if not client_ids:
        raise ValueError("client_ids must not be empty")
    groups = defaultdict(list)
    for cid in client_ids:
        groups[str(domain_ids.get(cid, cid))].append(cid)
    domain_weight = 1.0 / float(max(1, len(groups)))
    weights = {}
    for members in groups.values():
        per_client = domain_weight / float(max(1, len(members)))
        for cid in members:
            weights[cid] = per_client
    return weights


def aggregate_state_dicts(
    client_states: Mapping[str, Mapping[str, torch.Tensor]],
    client_num_samples: Mapping[str, int],
    exclude_keys: Set[str],
    agg_weight: str = "num_samples",
    domain_ids: Mapping[str, str] | None = None,
    client_weights: Mapping[str, float] | None = None,
):
    """Aggregate client state_dicts while skipping explicitly local keys."""
    if not client_states:
        raise ValueError("client_states must not be empty")
    client_ids = list(client_states.keys())
    if client_weights is not None:
        missing = [cid for cid in client_ids if cid not in client_weights]
        if missing:
            raise KeyError(f"client_weights missing clients: {missing}")
        total = float(sum(max(0.0, float(client_weights[cid])) for cid in client_ids))
        if total <= 0.0:
            raise ValueError("client_weights must sum to a positive value")
        weights = {cid: max(0.0, float(client_weights[cid])) / total for cid in client_ids}
    elif str(agg_weight) in {"domain_uniform", "domain_balanced"}:
        weights = domain_balanced_aggregation_weights(client_ids, domain_ids or {cid: cid for cid in client_ids})
    else:
        weights = _aggregation_weights(client_ids, client_num_samples, agg_weight)
    ref_state = client_states[client_ids[0]]
    excluded = set(exclude_keys or set())
    new_state = OrderedDict()

    for key, ref_tensor in ref_state.items():
        if key in excluded:
            continue
        for cid in client_ids:
            if key not in client_states[cid]:
                raise KeyError(f"Client {cid} is missing state key {key}")
            if client_states[cid][key].shape != ref_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: client {cid} has {tuple(client_states[cid][key].shape)}, "
                    f"expected {tuple(ref_tensor.shape)}"
                )

        if not torch.is_floating_point(ref_tensor):
            new_state[key] = ref_tensor.detach().clone()
            continue

        acc = torch.zeros_like(ref_tensor)
        for cid in client_ids:
            acc.add_(client_states[cid][key].detach().to(dtype=ref_tensor.dtype), alpha=float(weights[cid]))
        new_state[key] = acc

    return new_state


def resolve_exclude_keys(
    state: Mapping[str, torch.Tensor],
    *,
    exact_keys: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> Set[str]:
    """Resolve local-only state keys before FedAvg/FedProx aggregation."""
    exact = {str(k) for k in (exact_keys or []) if str(k)}
    prefix_tuple = tuple(str(p) for p in (prefixes or []) if str(p))
    out: Set[str] = set()
    for key in state.keys():
        if key in exact or (prefix_tuple and key.startswith(prefix_tuple)):
            out.add(key)
    return out
