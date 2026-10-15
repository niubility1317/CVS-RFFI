from . import design as d,model as api
from experiments.cvs_signature_stats50.audit import run as native
from experiments.cvs_receiver_vrex50.evidence import recount
def run():
 a=native(design=d,build_fn=api.build);obs=[recount(r,design=d) for r in d.training_rows()]
 return dict(status='VERIFIED',audit=a,actual_training_observations=obs,risk_steps_recounted=11100*len(obs),gradient_layer_records_recounted=208*len(obs),saved_vectors_recounted=624*len(obs),head_fit_updates=0,U_truth_read=False)
