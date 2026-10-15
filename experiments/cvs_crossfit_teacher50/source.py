"""Original native CE/cos optimizer path, with scratch RX exposure filter."""
from experiments.cvs_urban_sigtime50.source import train as original_train
from . import design,model,subset
from .observer import Observer


def train(c):
    observer=Observer(c)
    original_train(c,design=design,model_api=model,observer=observer,data_adapter=subset.install)
    return observer.finish()
