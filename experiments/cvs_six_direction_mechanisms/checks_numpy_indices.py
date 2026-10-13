"""Regression for Torch runtimes without NumPy integer dtype inference."""
import json
from unittest.mock import patch
import numpy as np
import torch
from .matching import partial_plan, validate_plan


def check():
    cost = torch.tensor([[.1, .7, .3, .9], [.2, .8, .4, .1],
                         [.9, .1, .8, .2], [.6, .3, .1, .4]], dtype=torch.float64)
    native = torch.as_tensor
    def without_numpy_inference(value, *args, **kwargs):
        if isinstance(value, (np.ndarray, np.generic)):
            raise RuntimeError('Could not infer dtype of numpy.int64')
        return native(value, *args, **kwargs)
    with patch('torch.as_tensor', side_effect=without_numpy_inference):
        plan = partial_plan(cost, 2)
    validate_plan(plan, 4, 2)
    assert abs(float((plan * cost).sum()) - .05) < 1e-12
    return dict(status='PASS', numpy_integer_bridge_unavailable=True,
                exact_pair_count=2, exact_mass=.5, constrained_optimum=.05)


if __name__ == '__main__':
    print(json.dumps(check()))
