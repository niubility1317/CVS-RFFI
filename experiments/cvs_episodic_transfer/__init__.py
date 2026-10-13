"""Temporary differentiable classifiers for source-only feature training."""
from .ridge import episode_loss, ridge_logits

__all__ = ["episode_loss", "ridge_logits"]
