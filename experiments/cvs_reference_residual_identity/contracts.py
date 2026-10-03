"""Deterministic architecture contract without allocating or seeding a model."""
from experiments.cvs_reference_residual_identity.model import reference_residual_contract
def architecture_contract(variant):
    return dict(reference_residual_contract(variant),total_parameters=189562,new_parameters=25337)
PRECISION='float32_parameters_backbone;float64_complex128_physical_operator'
