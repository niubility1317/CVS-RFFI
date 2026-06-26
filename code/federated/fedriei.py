from __future__ import annotations

"""Compatibility shim for FedRIEI paper-method primitives."""

from Fedbase.FedRIEI import (
    apply_fedriei_server_gradient_step,
    compress_gradient_tensor,
    compressed_gradient_from_states,
    fedriei_alternating_step,
    fedriei_loss_terms,
    normalize_compression_name,
)

__all__ = [
    "apply_fedriei_server_gradient_step",
    "compress_gradient_tensor",
    "compressed_gradient_from_states",
    "fedriei_alternating_step",
    "fedriei_loss_terms",
    "normalize_compression_name",
]
