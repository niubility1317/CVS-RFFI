"""Same source observer and native trajectory, with the declared held RX."""
from . import design as d,model as api
from experiments.cvs_ordinary_fold20.source import train as original


def train(c):return original(c,design=d,model_api=api)
