"""Same legal As20 student and frozen EMA teacher; no second teacher source."""
from . import design as d
from experiments.cvs_noise_shape_factorial43 import source as original
from experiments.cvs_crossview_supcon41 import source as backend


def parent_payload(c):
    d.validate(c)
    expected = d.old_config()
    if c['parent_source'] != expected['parent_source']:
        raise ValueError('Only own original P43 B01 As scratch20 parent')
    return original.parent_payload(expected)


def teacher_payload(c, arm='B01'):
    if arm != 'B01':
        raise ValueError('Only the same own As20 EMA teacher is registered')
    return parent_payload(c)


def segment_train(c, design=None, model_api=None, observer=None):
    previous = backend.parent_payload
    backend.parent_payload = parent_payload
    try:
        return backend.segment_train(c, design=design, model_api=model_api, observer=observer)
    finally:
        backend.parent_payload = previous
