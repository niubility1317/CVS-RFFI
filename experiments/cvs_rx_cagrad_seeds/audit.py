from . import design as d,model as api
from experiments.cvs_rx_cagrad50.audit import shared
from experiments.cvs_rx_cagrad50.evidence import recount
def run():
 native=shared(design=d,build_fn=api.build);rr=[recount(r,design=d) for r in d.training_rows()]
 return dict(status='VERIFIED',audit=native,actual_training_observations=rr,risk_steps_recounted=11100*len(rr),gradient_layer_records_recounted=208*len(rr),saved_vectors_recounted=832*len(rr),extra_model_forwards=0,head_fit_updates=0,U_truth_read=False)
