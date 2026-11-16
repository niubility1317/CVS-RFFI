import pytest
import torch

from experiments.pamf_repl_weight.source_adamw_diagnostic import capture_guard_derivatives, difference


def test_guard_derivatives_are_captured_and_original_function_restored():
    original = torch.autograd.grad
    x = torch.tensor([2.], requires_grad=True)
    with capture_guard_derivatives() as captured:
        torch.autograd.grad(x.square().sum(), [x], retain_graph=True)
        torch.autograd.grad((3*x).sum(), [x])
    assert torch.equal(captured[0][0], torch.tensor([4.]))
    assert torch.equal(captured[1][0], torch.tensor([3.]))
    assert torch.autograd.grad is original


def test_guard_observer_restores_autograd_after_error():
    original = torch.autograd.grad
    with pytest.raises(RuntimeError):
        with capture_guard_derivatives():
            raise RuntimeError('expected')
    assert torch.autograd.grad is original


def test_incremental_displacement_handles_disconnected_parameters():
    p = torch.nn.Parameter(torch.tensor([0., 0.]))
    named = [('a', p), ('b', p), ('c', p)]
    observed = difference((torch.tensor([2., 3.]), None, None),
                          (torch.tensor([1., 1.]), torch.tensor([4., 5.]), None), named)
    assert torch.equal(observed[0], torch.tensor([1., 2.]))
    assert torch.equal(observed[1], torch.tensor([-4., -5.]))
    assert observed[2] is None
