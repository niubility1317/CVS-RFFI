"""Paper-faithful FSCIL-SEI reproduction with orthogonal space constraints."""

from .metrics import average_incremental_metrics, harmonic_accuracy
from .model import CosineClassifier, SixBlockConv1DEncoder, class_mean_weights
from .pseudo_targets import assign_base_targets, make_simplex_pseudo_targets

__all__ = [
    "CosineClassifier",
    "SixBlockConv1DEncoder",
    "assign_base_targets",
    "average_incremental_metrics",
    "class_mean_weights",
    "harmonic_accuracy",
    "make_simplex_pseudo_targets",
]
