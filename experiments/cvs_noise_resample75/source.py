"""The one legal SAT18 AS_CE20 student/ownEMA parent, without another teacher."""
from . import design as d
from experiments.cvs_satellite_dropout18 import source as original
from experiments.cvs_crossview_supcon41 import source as backend


def parent_payload(c):
    d.validate(c)
    expected = d.old_config(c['arm'])
    if c['parent_source'] != expected['parent_source']:
        raise ValueError('Only the matching SAT18 AS_CE scratch20 parent')
    return original.parent_payload(expected)


def segment_train(c, design=None, model_api=None, observer=None):
    previous = backend.parent_payload
    backend.parent_payload = parent_payload
    try:
        return backend.segment_train(c, design=design, model_api=model_api, observer=observer)
    finally:
        backend.parent_payload = previous
