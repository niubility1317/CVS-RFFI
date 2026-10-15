from . import design as d,model
from experiments.cvs_feature_sequence50.source import train as shared_train
def train(c):
    d.validate(c)
    # No observer: no RX masking, auxiliary loss, probe update or held-source read.
    return shared_train(c,design=d,model_api=model)
