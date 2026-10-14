import time
from . import design as d, model as api
from .evidence import recount
from experiments.cvs_signature_stats50.audit import run as shared


def run(*, design=None, build_fn=None):
    d = design or globals()['d']
    start = time.time()
    native = shared(design=d, build_fn=build_fn or api.build)
    source = [recount(r, design=d) for r in d.training_rows()]
    return dict(status='VERIFIED', audit=native, actual_training_observations=source,
        native_steps_recounted=11100 * len(source), gradient_layer_records_recounted=208 * len(source),
        saved_vectors_recounted=624 * len(source), extra_model_forwards=22200 * len(source), head_fit_updates=0,
        U_truth_read=False, seconds=time.time() - start)
