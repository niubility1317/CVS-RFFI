from . import design as d
from .predict import source_proof
from experiments.cvs_sinc_units50.evidence import recount as shared
def recount(row):return shared(row,design=d,proof_fn=source_proof)
