"""Verify each untouched P49 scratch parent before loading its student/EMA."""
from . import design as d
from experiments.cvs_As_replicate49 import source as parent_source
from experiments.cvs_crossview_supcon41 import source as backend


def parent_payload(c):
    d.validate(c)
    expected = d.parent_config(c)
    if c['parent_source'] != expected['parent_source']:
        raise ValueError('Only corresponding original As20 parent allowed')
    cp = parent_source.parent_payload(expected)
    if cp['args']['seed'] != c['model_seed']:
        raise ValueError('Parent actual native model seed differs')
    return cp


def segment_train(c, design=None, model_api=None, observer=None):
    previous = backend.parent_payload
    backend.parent_payload = parent_payload
    try:
        return backend.segment_train(c, design=design, model_api=model_api, observer=observer)
    finally:
        backend.parent_payload = previous
