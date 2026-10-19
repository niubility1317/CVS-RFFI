from . import design as d,model as api
from experiments.cvs_F_nontrue20.observer import Observer
from experiments.cvs_feature_sequence50.source import train as shared_train
def train(c):
 o=Observer(c);shared_train(c,design=d,model_api=api,observer=o);return o.finish()
