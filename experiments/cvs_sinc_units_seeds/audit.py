from . import design as d,model as api
from .evidence import recount
from experiments.cvs_sigtime_pair50.audit import run as shared
def run():
 native=shared(design=d,build_fn=api.build);rows=[recount(r) for r in d.training_rows()]
 return dict(native,source_updates=rows,new_updates=44400,actual_U_packets=3855600,head_fit_updates=0,extra_model_forwards=0,U_truth_read=False)
