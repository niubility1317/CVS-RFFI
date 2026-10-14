from . import design as d, model as api
from .predict import source_proof
from .feature_check import run
from experiments.cvs_receiver_dro50.source import train as shared


def train(c):
    d.require_source()
    return shared(c, design=d, model_api=api, proof_fn=source_proof, feature_fn=run)
