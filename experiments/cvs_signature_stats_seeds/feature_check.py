from . import design as d
from experiments.cvs_signature_stats50.feature_check import run as shared
def run(seed=None,only_row=None):return shared(seed,only_row,design=d)
