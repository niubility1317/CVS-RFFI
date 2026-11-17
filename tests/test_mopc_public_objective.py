import torch
import torch.nn.functional as F

from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import _mopc_public_objective


def test_author_loss_and_gradients_use_temperature_two_and_unsquared_norm():
    torch.manual_seed(11)
    logits = torch.randn(4, 3, requires_grad=True)
    proto = torch.randn(4, 3, requires_grad=True)
    params = {'a': torch.randn(3, 2, requires_grad=True), 'b': torch.randn(3, requires_grad=True)}
    previous = {name: torch.randn_like(value) for name, value in params.items()}
    labels = torch.tensor([0, 1, 2, 0]); proto_labels = torch.tensor([1, 2, 0, 1])
    reg = sum(.7 * (1 - j / len(params)) * (p - previous[name]).norm(2)
              for j, (name, p) in enumerate(params.items()))
    expected = F.cross_entropy(logits, labels) + F.cross_entropy(proto / 2, proto_labels) + reg
    actual = _mopc_public_objective(logits, labels, proto, proto_labels, params, previous, lambda_max=.7)
    variables = [logits, proto, *params.values()]
    expected_grad = torch.autograd.grad(expected, variables, retain_graph=True)
    actual_grad = torch.autograd.grad(actual.total, variables)
    assert torch.equal(expected, actual.total)
    assert all(torch.equal(a, b) for a, b in zip(expected_grad, actual_grad))
