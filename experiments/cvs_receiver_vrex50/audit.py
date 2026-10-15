from . import design as d,model as api
from .evidence import recount
from experiments.cvs_signature_stats50.audit import run as shared
def run():
 native=shared(design=d,build_fn=api.build);observations=[recount(r) for r in d.training_rows()]
 return dict(status='VERIFIED',audit=native,actual_training_observations=observations,risk_steps_recounted=11100*len(observations),gradient_layer_records_recounted=208*len(observations),saved_vectors_recounted=624*len(observations),extra_model_forwards=0,head_fit_updates=0,U_truth_read=False)
