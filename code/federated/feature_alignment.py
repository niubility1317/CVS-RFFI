from __future__ import annotations

"""Compatibility shim for FedFA paper-method primitives."""

from Fedbase.FedFA import (
    ComplexConv1d,
    ComplexConvBlock1d,
    FedFAComplexCNN,
    covariance_matrix,
    pairwise_coral_alignment_loss,
    peer_coral_alignment_losses,
)

__all__ = [
    "ComplexConv1d",
    "ComplexConvBlock1d",
    "FedFAComplexCNN",
    "covariance_matrix",
    "pairwise_coral_alignment_loss",
    "peer_coral_alignment_losses",
]
