import math
import time

import pytest
import torch

from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import _record


def test_finite_gradient_with_fp32_norm_overflow_is_recorded_without_changing_update():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.zero_()
    model.weight.grad = torch.full_like(model.weight, 1e20)
    before = model.weight.grad.clone()
    assert not torch.isfinite(model.weight.grad.norm())
    trace = []
    _record(trace, model, {'loss': torch.tensor(1.)}, None, time.perf_counter(), method='mopc_hr')
    assert math.isfinite(trace[0]['gradient_norm'])
    torch.testing.assert_close(model.weight.grad, before, rtol=0, atol=0)
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert torch.isfinite(model.weight).all()
    torch.testing.assert_close(model.weight, -before * .01)


@pytest.mark.parametrize('value', [float('inf'), float('nan')])
def test_nonfinite_gradient_still_stops(value):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, value)
    with pytest.raises(FloatingPointError, match='gradient_norm'):
        _record([], model, {'loss': torch.tensor(1.)}, None, time.perf_counter(), method='mopc_hr')
