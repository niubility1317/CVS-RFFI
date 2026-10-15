"""Original native optimizer/EMA/U path; new views remain real labeled IQ."""
from . import design,model
from .observer import Observer
from experiments.cvs_urban_sigtime50.source import train as original_train
from experiments.cvs_urban_semantic50.source import adapter


def train(c):
    observer=Observer(c)
    original_train(c,design=design,model_api=model,observer=observer,data_adapter=adapter)
    return observer.finish()
