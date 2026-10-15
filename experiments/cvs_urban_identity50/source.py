"""Original joint trainer, with one replacement auxiliary objective."""
from experiments.cvs_urban_sigtime50.source import train as original_train
from . import design,model
from .observer import Observer


def train(c):
    observer=Observer(c)
    original_train(c,design=design,model_api=model,observer=observer)
    return observer.finish()
