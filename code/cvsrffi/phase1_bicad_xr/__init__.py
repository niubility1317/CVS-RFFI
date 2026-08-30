"""Public Task1 configuration API for the ADV3B02-BiCAD-XR candidate family."""

from .config import (
    BiCADXRConfig,
    BiCADXRStage,
    CANDIDATE_IDS,
    LEO_WEAK_SCENARIOS,
    candidate_config,
    candidate_diff,
    stage_for_update,
)

__all__ = [
    "BiCADXRConfig",
    "BiCADXRStage",
    "CANDIDATE_IDS",
    "LEO_WEAK_SCENARIOS",
    "candidate_config",
    "candidate_diff",
    "stage_for_update",
]
