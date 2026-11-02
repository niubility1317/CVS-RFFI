"""DURA-aligned spectral fusion preserving unavailable boundaries."""
from experiments.phase1_dura_blendgate.model import make_model as make_reflection_model


def make_model():
    model = make_reflection_model()
    model.preserve_edges = True
    return model
