from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,finish as shared_finish
def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():return shared_finish(design=d)
