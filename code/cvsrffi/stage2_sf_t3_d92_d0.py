"""Support-only D0/H6 Compact ``t3.norm`` delta for fixed D92-E0.

The H6 Compact suffix uses a target head while fitting support.  That head is
temporary for this candidate: only the two ``t3.norm`` affine deltas cross the
persistence boundary.  The returned candidate specification is intentionally
query-free and locks the downstream registered-class head to D92-E0-NORF32.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch


SCHEMA = "cvs.stage2.sf_t3_d92_d0.v1"
CANDIDATE_ID = "D0_H6_COMPACT_T3_D92_E0"
METHOD_LOCK = "D92-E0-NORF32"
PERSISTED_PARAMETER_NAMES = (
    "model.t3.norm.weight",
    "model.t3.norm.bias",
)


class D0H6CompactDeltaError(RuntimeError):
    """Raised when the support-only compact-delta boundary drifts."""


@dataclass(frozen=True)
class D0H6CompactT3NormDelta:
    """The only persistent model delta plus its support-only audit."""

    model_deltas: Mapping[str, torch.Tensor]
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = {
            name: value.detach().cpu().clone()
            for name, value in self.model_deltas.items()
        }
        object.__setattr__(self, "model_deltas", MappingProxyType(copied))
        object.__setattr__(self, "audit", MappingProxyType(dict(self.audit)))


def build_d0_h6_compact_candidate_spec() -> Mapping[str, Any]:
    """Return the fixed, runner-facing candidate contract."""

    return MappingProxyType(
        {
            "schema": SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "sf_tapft_row": "D0",
            "adapter_execution": "H6_COMPACT",
            "persistent_parameter_names": PERSISTED_PARAMETER_NAMES,
            "temporary_target_head_policy": "discard_after_support_fit",
            "method_lock": METHOD_LOCK,
            "d92_enabled": True,
            "e0_locked": True,
            "rf32_used": False,
            "support_only": True,
            "query_rows_used": 0,
        }
    )


def _floating_tensor(value: Any, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.numel() == 0
        or not bool(torch.isfinite(value).all())
    ):
        raise D0H6CompactDeltaError(
            f"{name} must be a nonempty finite floating tensor"
        )
    return value.detach().cpu()


def _base_parameter(
    state: Mapping[str, Any], canonical_name: str
) -> torch.Tensor:
    short_name = canonical_name.removeprefix("model.")
    if canonical_name in state:
        return _floating_tensor(state[canonical_name], name=canonical_name)
    if short_name in state:
        return _floating_tensor(state[short_name], name=short_name)
    raise D0H6CompactDeltaError(f"missing base parameter: {canonical_name}")


def build_support_only_t3_norm_delta(
    base_model_state: Mapping[str, Any],
    compact_adapted_state: Mapping[str, Any],
    *,
    support_rows_used: int,
) -> D0H6CompactT3NormDelta:
    """Filter one D0/H6 Compact fit into the two persistent affine deltas.

    No query object is accepted by this interface.  ``compact_adapted_state``
    may contain the temporary target head exported by ``CompactH6Suffix``;
    every such entry is recorded as discarded and never returned.
    """

    if not isinstance(base_model_state, Mapping) or not isinstance(
        compact_adapted_state, Mapping
    ):
        raise D0H6CompactDeltaError("base and adapted states must be mappings")
    if (
        isinstance(support_rows_used, bool)
        or not isinstance(support_rows_used, int)
        or support_rows_used <= 0
    ):
        raise D0H6CompactDeltaError("support_rows_used must be a positive integer")

    deltas: dict[str, torch.Tensor] = {}
    for name in PERSISTED_PARAMETER_NAMES:
        if name not in compact_adapted_state:
            raise D0H6CompactDeltaError(f"missing adapted parameter: {name}")
        adapted = _floating_tensor(compact_adapted_state[name], name=name)
        base = _base_parameter(base_model_state, name)
        if adapted.shape != base.shape:
            raise D0H6CompactDeltaError(f"parameter shape drift: {name}")
        deltas[name] = adapted - base.to(dtype=adapted.dtype)

    discarded_heads = tuple(
        sorted(
            str(name)
            for name in compact_adapted_state
            if str(name).startswith(("head.", "target_head."))
        )
    )
    spec = build_d0_h6_compact_candidate_spec()
    audit = {
        **dict(spec),
        "support_rows_used": support_rows_used,
        "query_rows_used": 0,
        "persisted_parameter_names": PERSISTED_PARAMETER_NAMES,
        "temporary_target_head_discarded": bool(discarded_heads),
        "discarded_target_head_names": discarded_heads,
    }
    return D0H6CompactT3NormDelta(model_deltas=deltas, audit=audit)


__all__ = [
    "CANDIDATE_ID",
    "D0H6CompactDeltaError",
    "D0H6CompactT3NormDelta",
    "METHOD_LOCK",
    "PERSISTED_PARAMETER_NAMES",
    "SCHEMA",
    "build_d0_h6_compact_candidate_spec",
    "build_support_only_t3_norm_delta",
]
