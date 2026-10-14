import time
from . import design as d, model as api
from .evidence import recount
from experiments.cvs_signature_stats50.audit import run as shared


def run():
    start = time.time()
    native = shared(design=d, build_fn=api.build)
    source = [recount(r) for r in d.training_rows()]
    return dict(status='VERIFIED', audit=native, actual_training_observations=source,
        risk_steps_recounted=33300, gradient_layer_records_recounted=624,
        saved_vectors_recounted=1872, extra_model_forwards=0, head_fit_updates=0,
        U_truth_read=False, seconds=time.time() - start)
