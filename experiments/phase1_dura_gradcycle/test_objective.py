import unittest
import torch
from torch import nn

from experiments.phase1_dura_gradcycle.objective import balanced_objective


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_output = nn.Linear(2, 1, bias=False)


class GradientBalanceTest(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.weight = self.model.gate_output.weight

    def test_opposing_component_is_projected_and_remainder_balanced(self):
        main = self.weight[0, 0]
        auxiliary = -self.weight[0, 0] + self.weight[0, 1]
        loss = balanced_objective(main, auxiliary, self.model)
        gradient = torch.autograd.grad(loss, self.weight)[0]
        torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.2]]), rtol=1e-5, atol=1e-6)

    def test_nonconflicting_auxiliary_has_bounded_strength(self):
        main = self.weight[0, 0]
        auxiliary = self.weight[0, 1]
        gradient = torch.autograd.grad(balanced_objective(main, auxiliary, self.model), self.weight)[0]
        torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.2]]), rtol=1e-5, atol=1e-6)

    def test_fully_opposed_auxiliary_does_not_reverse_main(self):
        main = self.weight[0, 0]
        auxiliary = -self.weight[0, 0]
        gradient = torch.autograd.grad(balanced_objective(main, auxiliary, self.model), self.weight)[0]
        torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.0]]))

    def test_scale_cap_limits_near_zero_auxiliary(self):
        main = self.weight[0, 0]
        auxiliary = 1e-7 * self.weight[0, 1]
        gradient = torch.autograd.grad(balanced_objective(main, auxiliary, self.model), self.weight)[0]
        torch.testing.assert_close(gradient, torch.tensor([[1.0, 5e-6]]), rtol=1e-5, atol=1e-8)

    def test_zero_main_gradient_keeps_auxiliary(self):
        main = (self.weight * 0).sum()
        auxiliary = self.weight[0, 1]
        gradient = torch.autograd.grad(balanced_objective(main, auxiliary, self.model), self.weight)[0]
        torch.testing.assert_close(gradient, torch.tensor([[0.0, 1.0]]))


if __name__ == '__main__': unittest.main()
