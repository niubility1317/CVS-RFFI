"""CVS-aligned evaluation and adaptation layers for paper baselines."""

from .supervised_da import (
    dadda_sda_objective,
    mrior_sda_objective,
    validate_supervised_da_manifest,
)

__all__ = [
    "dadda_sda_objective",
    "mrior_sda_objective",
    "validate_supervised_da_manifest",
]
