"""Load only corresponding exact legal P43 own20 student, EMA and AdamW payload."""
from . import design as d
from .optimizer_state import validate_saved
from experiments.cvs_noise_shape_factorial43 import source as original
from experiments.cvs_crossview_supcon41 import source as backend


def parent_payload(c):
    d.validate(c)
    expected = d.old_config(c['arm'])
    if c['parent_source'] != expected['parent_source']:
        raise ValueError('Only corresponding own P43 scratch20 parent')
    cp = original.parent_payload(expected)
    validate_saved(cp.get('optimizer'), expected_steps=4440)
    return cp


def segment_train(c, design=None, model_api=None, observer=None):
    previous = backend.parent_payload
    backend.parent_payload = parent_payload
    try:
        return backend.segment_train(c, design=design, model_api=model_api, observer=observer)
    finally:
        backend.parent_payload = previous
