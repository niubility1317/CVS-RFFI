from . import design as d
from .predict import source_proof,own_model
from experiments.cvs_identity_mixstyle50.feature_check import run as shared_run

def run(seed=None):return shared_run(seed,design=d,proof_fn=source_proof,build_fn=own_model)
