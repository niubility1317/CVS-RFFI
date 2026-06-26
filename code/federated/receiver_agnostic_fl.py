from __future__ import annotations

"""Compatibility shim for RAFL paper-method primitives."""

from Fedbase.RAFL import (
    aggregate_label_losses,
    label_loss_driven_client_selection,
    receiver_agnostic_loss,
)

__all__ = [
    "aggregate_label_losses",
    "label_loss_driven_client_selection",
    "receiver_agnostic_loss",
]
