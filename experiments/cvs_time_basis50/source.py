from . import design as d,model
from experiments.cvs_feature_sequence50.source import train as shared
from experiments.cvs_time_factorial50.observe import Observer
def train(c):return shared(c,design=d,model_api=model,observer=Observer())
