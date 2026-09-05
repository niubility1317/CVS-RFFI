"""Observation packets for a new pair run; no historical P5 causal claim."""
from __future__ import annotations

from collections.abc import Mapping
import copy
import math
import numbers
from typing import Any

import torch


_TRUTH_FIELDS = frozenset({
    "truth", "true_tx_i", "tx_i", "tx_label", "tx_labels", "y_tx",
    "u_truth", "u_labels", "unlabeled_truth", "unlabeled_labels",
    "target_truth", "target_labels", "query_truth", "query_labels",
})


def _reject_truth_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _TRUTH_FIELDS:
                raise ValueError("failure packet must not contain TX/query truth fields")
            _reject_truth_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_truth_fields(child)


def _snapshot(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(child) for child in value)
    if isinstance(value, list):
        return [_snapshot(child) for child in value]
    return copy.deepcopy(value)


def _all_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(child) for child in value)
    if isinstance(value, numbers.Number):
        return math.isfinite(value)
    return True


def _validate_physical_ids(value: Any) -> None:
    if not isinstance(value, Mapping) or any(key not in {"l", "u", "labeled", "unlabeled"} for key in value):
        raise ValueError("source physical IDs require labeled/unlabeled groups")
    for ids in value.values():
        if torch.is_tensor(ids):
            if ids.dtype == torch.bool or ids.is_floating_point() or ids.ndim not in (1, 2):
                raise ValueError("source physical IDs must be integer identities")
            if ids.ndim == 2 and ids.shape[1] != 5:
                raise ValueError("source physical ID metadata must have five fields")
            continue
        if not isinstance(ids, (list, tuple)):
            raise ValueError("source physical IDs must be identity sequences")
        for identity in ids:
            if isinstance(identity, (str, numbers.Integral)):
                continue
            if isinstance(identity, (list, tuple)) and len(identity) == 5 and all(isinstance(item, numbers.Integral) for item in identity):
                continue
            raise ValueError("source physical IDs must be opaque IDs or five metadata fields")


def build_pair_failure_payload(*, epoch: int, batch: int, loss: Any,
        labeled_components: Mapping[str, Any], unlabeled_components: Mapping[str, Any],
        first_nonfinite_gradient: Any, args: Any,
        source_physical_ids: Mapping[str, Any], gradscale: Any = None,
        **optional_payload: Any) -> dict[str, Any]:
    """Build one immutable-in-practice CPU snapshot; caller owns first-write gating.

    IDs are physical metadata identities, never TX labels. Optional payload may
    include RNG, model/EMA, optimizer/scaler states and operator diagnostics.
    Named truth fields are rejected; callers must also avoid disguising truth
    as opaque IDs. This helper does not read a dataset, rerun a batch or write.
    """
    _validate_physical_ids(source_physical_ids)
    args_dict = dict(args) if isinstance(args, Mapping) else vars(args)
    payload = {
        "schema": "cvs.pair_reform.first_anomaly.v1",
        "claim_boundary": "OBSERVATION_ONLY_NOT_P5_ROOT_CAUSE",
        "epoch": int(epoch), "batch_index": int(batch),
        "loss": loss,
        "labeled_components": dict(labeled_components),
        "unlabeled_components": dict(unlabeled_components),
        "first_nonfinite_gradient": first_nonfinite_gradient,
        "args": args_dict, "source_physical_ids": source_physical_ids,
        "gradscale": gradscale,
    }
    if set(optional_payload) & (set(payload) | {"finite"}):
        raise ValueError("optional payload must not replace reserved evidence fields")
    payload.update(optional_payload)
    _reject_truth_fields(payload)
    payload = _snapshot(payload)
    payload["finite"] = {
        "loss": _all_finite(payload["loss"]),
        "labeled_components": {key: _all_finite(value) for key, value in payload["labeled_components"].items()},
        "unlabeled_components": {key: _all_finite(value) for key, value in payload["unlabeled_components"].items()},
        "gradscale": _all_finite(payload["gradscale"]),
    }
    return payload
