"""Rollback of optimizer transactions, including stochastic forward state."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RNGState:
    python: object
    numpy: tuple
    cpu: torch.Tensor
    cuda: list

    @classmethod
    def capture(cls):
        return cls(random.getstate(), copy.deepcopy(np.random.get_state()),
                   torch.get_rng_state().clone(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda:
            torch.cuda.set_rng_state_all(self.cuda)


def clone_buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def restore_buffers(model, buffers):
    with torch.no_grad():
        current = dict(model.named_buffers())
        if current.keys() != buffers.keys():
            raise RuntimeError("A closure changed the registered buffer structure")
        for name, value in buffers.items():
            current[name].copy_(value)


class TrainingState:
    """An in-memory transaction snapshot, not a checkpoint serialization format.

    Mutable loss state must be supplied through ``stateful`` or kept outside the
    closure and committed by the caller after an accepted step. ``scaler`` is
    snapshotted when supplied; solvers scale autograd and then unscale gradients.
    """

    def __init__(self, model, optimizer, *, scaler=None, stateful=()):
        self.model, self.optimizer, self.scaler = model, optimizer, scaler
        self.parameters = [(p, p.detach().clone(), None if p.grad is None else p.grad.detach().clone())
                           for p in model.parameters()]
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        self.buffers = clone_buffers(model)
        self.rng = RNGState.capture()
        self.scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
        self.stateful = [(obj, copy.deepcopy(obj.state_dict())) for obj in stateful]

    def restore(self, *, rng=True, buffers=True):
        with torch.no_grad():
            for parameter, value, gradient in self.parameters:
                parameter.copy_(value)
                parameter.grad = None if gradient is None else gradient.clone()
        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        if buffers:
            restore_buffers(self.model, self.buffers)
        if rng:
            self.rng.restore()
        if self.scaler is not None:
            self.scaler.load_state_dict(copy.deepcopy(self.scaler_state))
        for obj, state in self.stateful:
            obj.load_state_dict(copy.deepcopy(state))


StateSnapshot = TrainingState
