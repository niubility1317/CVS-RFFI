"""Regression for the N607 NumPy/Torch bridge failure, without target access."""
import json
from unittest.mock import patch

import numpy as np
import torch

from .evaluate import iq_tensor


def check():
    arrays = [np.arange(n * 512, dtype=np.float32).reshape(n, 2, 256) / 131.0
              for n in (1, 64, 256)]
    maximum = 0.0
    for values in arrays:
        expected = torch.tensor(values.tolist(), dtype=torch.float32)
        # Explicitly emulate the failing installed bridge; the supported route
        # must never call from_numpy or Tensor.numpy.
        with patch.object(torch, 'from_numpy', side_effect=TypeError('expected np.ndarray (got numpy.ndarray)')):
            actual = iq_tensor(values, 'cpu')
            assert torch.equal(actual, expected)
            maximum = max(maximum, (actual - expected).abs().max().item())
            # Tensor owns writable copied storage, never the mmap input.
            actual.fill_(0)
            assert values.flat[-1] != 0
    try:
        iq_tensor(np.zeros((1, 2, 256), dtype=np.float64), 'cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong dtype was accepted')
    return dict(status='PASS', batches=[1, 64, 256], max_abs_error=maximum,
                simulated_numpy_bridge_unavailable=True, target_access=False)


if __name__ == '__main__':
    print(json.dumps(check()))
