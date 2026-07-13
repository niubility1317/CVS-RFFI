"""Paper-faithful core operations for MoPC-HR."""

from .algorithm import (
    compute_class_prototypes,
    correct_old_prototypes,
    hierarchical_regularization,
    mopc_hr_incremental_objective,
    prototype_augmentation,
)
from .protocol import validate_mopc_hr_config

__all__ = [
    "compute_class_prototypes",
    "correct_old_prototypes",
    "hierarchical_regularization",
    "mopc_hr_incremental_objective",
    "prototype_augmentation",
    "validate_mopc_hr_config",
]
