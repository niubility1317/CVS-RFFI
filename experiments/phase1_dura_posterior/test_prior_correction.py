import unittest
import types
from unittest.mock import patch

import torch

from experiments.phase1_dura_posterior.prior_correction import PosteriorPriorCorrection
from experiments.phase1_dura_posterior.integrated import PosteriorInDura


class PriorCorrectionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(128)
        self.received = torch.randn(3, 2, 256)
        self.current = torch.randn(3, 2, 256)
        self.prior = torch.randn(3, 2, 256)
        self.posterior = torch.randn(3, 2, 256)
        self.log_variance = torch.randn(3)
        self.model = PosteriorPriorCorrection()

    def inputs(self):
        return self.received, self.current, self.prior, self.posterior, self.log_variance

    def test_initialization_preserves_original_prior_and_is_trainable(self):
        output = self.model(*self.inputs())
        self.assertTrue(torch.equal(output, self.prior))
        (output - self.posterior).square().mean().backward()
        grad = self.model.coefficients.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_phase_rotation_commutes_with_correction(self):
        torch.nn.init.normal_(self.model.coefficients.weight, std=0.1)
        torch.nn.init.normal_(self.model.coefficients.bias, std=0.1)
        angle = torch.tensor(0.73)
        c, s = angle.cos(), angle.sin()

        def rotate(x):
            return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), dim=1)

        expected = rotate(self.model(*self.inputs()))
        actual = self.model(*(rotate(x) for x in self.inputs()[:4]), self.log_variance)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_rejects_incompatible_posterior_shape(self):
        with self.assertRaises(ValueError):
            self.model(self.received, self.current, self.prior, self.posterior[:, :, :-1], self.log_variance)

    def test_both_integrated_stages_preserve_reference_and_receive_gradients(self):
        def pair(z):
            return torch.stack((z.real, z.imag), 1)

        def complex_iq(x):
            return torch.complex(x[:, 0], x[:, 1])

        class Action:
            def __call__(self, x): return x
            def adjoint(self, x): return x

        class Prior(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.factor = torch.nn.Parameter(torch.tensor(0.3))

            def forward(self, x):
                return self.factor*x[:, 4:6]

        class Dura(torch.nn.Module):
            arm = 'stage2'

            def __init__(self):
                super().__init__()
                self.prior = torch.nn.ModuleList((Prior(), Prior()))

            def estimate_context(self, x):
                return x*0.7, Action(), torch.ones(len(x))

        def weighted_map(action, observed, prior, variance, *, steps, initial):
            self.assertEqual(steps, 2)
            return (prior+initial+observed)/3

        modules = {
            'experiments.pamf_action_restore.models': types.SimpleNamespace(SCALE=0.5),
            'experiments.pamf_restoration.models': types.SimpleNamespace(pair=pair, complex_iq=complex_iq),
            'experiments.pamf_uncertainty_restore.models': types.SimpleNamespace(weighted_map=weighted_map),
        }
        dura = Dura()
        integrated = PosteriorInDura(dura).train()
        base = self.received*0.7
        expected = base
        for core in dura.prior:
            prior = base+0.5*core(torch.cat((self.received, base, expected,
                                            self.received, torch.zeros(3, 1, 256)), 1))
            expected = pair(weighted_map(Action(), complex_iq(self.received), complex_iq(prior),
                                         torch.ones(3, 256), steps=2, initial=complex_iq(expected)))
        with patch.dict('sys.modules', modules):
            actual = integrated(self.received, self.posterior, self.log_variance)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (actual-self.posterior).square().mean().backward()
        self.assertFalse(dura.training)
        self.assertTrue(all(p.grad is None for p in dura.parameters()))
        for correction in integrated.corrections:
            grad = correction.coefficients.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
