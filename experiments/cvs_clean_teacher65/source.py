"""All students inherit own As20; both frozen teachers have exact own20 provenance."""
from . import design as d
from experiments.cvs_noise_shape_factorial43 import source as original
from experiments.cvs_crossview_supcon41 import source as backend

def parent_payload(c):
    d.validate(c)
    expected = d.old_config()
    if c['parent_source'] != expected['parent_source']:
        raise ValueError('Only own original P43 B01 As scratch20 parent')
    return original.parent_payload(expected)

def teacher_payload(c, arm):
    d.validate(c)
    if arm not in ('B00', 'B01'):
        raise ValueError('Only same-seed own NoAs20/As20 EMA teacher sources')
    return original.parent_payload(d.reference_config(arm))

def segment_train(c, design=None, model_api=None, observer=None):
    previous = backend.parent_payload
    backend.parent_payload = parent_payload
    try:
        return backend.segment_train(c, design=design, model_api=model_api, observer=observer)
    finally:
        backend.parent_payload = previous
