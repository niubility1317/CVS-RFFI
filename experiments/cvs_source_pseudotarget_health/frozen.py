"""Exact identity state and RNG preservation around source-only diagnostics."""
from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def preserve_rng(device):
    python_state = random.getstate(); numpy_state = np.random.get_state()
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)


def snapshot(model):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('Every identity module must be in evaluation mode')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Identity parameters must be frozen without gradients')
    return dict(state={name: value.detach().clone() for name, value in model.state_dict().items()},
                modes=[module.training for module in model.modules()])


def assert_unchanged(model, before):
    current = model.state_dict()
    if current.keys() != before['state'].keys() or any(
            not torch.equal(value, current[name]) for name, value in before['state'].items()):
        raise ValueError('Diagnostic changed identity parameter or buffer')
    if before['modes'] != [module.training for module in model.modules()]:
        raise ValueError('Diagnostic changed identity module modes')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Diagnostic created identity gradients')
    return dict(parameters_buffers_exactly_unchanged=True, modes_unchanged=True,
                identity_gradient_count=0, identity_updates=0)
