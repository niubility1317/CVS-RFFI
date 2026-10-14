from . import design as d,model
from .observe import Observer
from experiments.cvs_feature_sequence50.source import train as shared

def train(c):return shared(c,design=d,model_api=model,observer=Observer())
