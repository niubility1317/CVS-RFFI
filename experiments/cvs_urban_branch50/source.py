"""Native E50 trainer with a declared L-only developer fold and graph-preserving hook."""
from experiments.cvs_urban_sigtime50.source import train as original_train
from . import design,model,inner_source
from .observer import Observer


def train(c):
    observer=Observer(c)
    original_train(c,design=design,model_api=model,observer=observer,data_adapter=inner_source.install)
    return observer.finish()
