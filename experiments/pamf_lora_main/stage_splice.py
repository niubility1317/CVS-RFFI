"""Separate LoRa coarse proposals and learned refinement using frozen weights."""

import torch
from torch.func import functional_call


def refinement_state(model):
    """Only the learned refinement network is exchanged."""
    return {f'prior.{key}': value for key, value in model.prior.state_dict().items()}


def read_stages(observed, rotation, quality, controls):
    direction, lag = controls
    args = (observed, quality, rotation)
    direction_coarse, direction_final = direction(*args)
    lag_coarse, lag_final = lag(*args)
    _, direction_to_lag = functional_call(direction, refinement_state(lag), args)
    _, lag_to_direction = functional_call(lag, refinement_state(direction), args)
    fine = tuple(torch.lerp(observed, output, .5)
                 for output in (direction_final, lag_final, direction_to_lag, lag_to_direction))
    return (direction_coarse, lag_coarse, *fine)
