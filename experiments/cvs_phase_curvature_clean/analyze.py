"""Independent same-row clean metric validation for the curvature family."""
from experiments.cvs_moment_residual_clean.analyze import (
    BASELINES, SEEDS, validate_results as _validate_results,
)
from experiments.cvs_phase_curvature_identity.model import VARIANTS

def validate_results(data, selection, previous):
    if selection.get('selected_variant') not in VARIANTS:
        raise ValueError('Unregistered phase-curvature candidate')
    return _validate_results(data, selection, previous)
