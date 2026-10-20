from functools import lru_cache
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_crossview_supcon41 import source as backend


@lru_cache(maxsize=1)
def original_parent():
    from experiments.cvs_as_temperature84 import model as old_model
    # Existing checker verifies the complete scratch20 -> own10 chain and roles.
    m = old_model.own_model(d.PARENT_CONFIG, 'cpu')
    del m
    cp = torch.load(d.PARENT_SOURCE / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if cp['args']['seed'] != d.SEED or cp.get('ema_model') is None:
        raise ValueError('Original T1 seed or own EMA differs')
    return cp


def parent_payload(c):
    d.validate(c)
    if c['arm'] != 'E30PLUS20':
        raise ValueError('Scratch row cannot read a training parent')
    return original_parent()


def segment_train(c, design=None, model_api=None, observer=None):
    previous = backend.parent_payload
    backend.parent_payload = parent_payload
    try:
        return backend.segment_train(c, design=design, model_api=model_api, observer=observer)
    finally:
        backend.parent_payload = previous
