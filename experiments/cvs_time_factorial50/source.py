from . import design as d,model
from .observe import Observer
from experiments.cvs_feature_sequence50.source import train as shared
from experiments.cvs_feature_sequence50.source import require_budget,native_default_head_only
def train(c):return shared(c,design=d,model_api=model,observer=Observer())
