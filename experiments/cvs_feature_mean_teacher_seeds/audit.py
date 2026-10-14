from . import design as d,model as api
from experiments.cvs_feature_mean_teacher50.audit import run as shared
def run():return shared(design=d,build_fn=api.build)
