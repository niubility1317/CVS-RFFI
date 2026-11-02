import unittest

import torch
from torch import nn

from experiments.phase1_dura_posterior.pipeline import GatedPosteriorDura, PhysicalPosterior


class Gate(nn.Module):
    def forward(self, x):
        return x[:, 0, 0]


class Dura(nn.Module):
    def forward(self, x):
        return x*0.5


class Integrated(nn.Module):
    def __init__(self):
        super().__init__()
        self.dura = Dura()
        self.factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, posterior, logvar):
        return self.dura(x) + self.factor*(posterior-x)


class Physical(nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, x):
        self.count += len(x)
        return x+1, torch.zeros(len(x), device=x.device)


class Blind(nn.Module):
    def forward(self, x):
        return torch.ones(len(x), 1, device=x.device), torch.zeros(len(x), device=x.device)


class PipelineTest(unittest.TestCase):
    def test_inactive_samples_keep_dura_and_physical_runs_only_for_active(self):
        x = torch.zeros(3, 2, 256)
        x[:, 0, 0] = torch.tensor([-3., 2., 3.])
        physical = Physical()
        model = GatedPosteriorDura(Integrated(), Gate(), physical).train()
        out, active = model(x)
        self.assertEqual(active.tolist(), [False, True, True])
        self.assertEqual(physical.count, 2)
        torch.testing.assert_close(out[0], x[0]*0.5, rtol=0, atol=0)
        torch.testing.assert_close(out[1:], x[1:]*0.5+0.1, rtol=0, atol=1e-7)
        out.sum().backward()
        self.assertGreater(float(model.integrated.factor.grad), 0)
        self.assertFalse(model.gate.training)
        self.assertFalse(model.posterior.training)

    def test_empty_gate_does_not_call_posterior(self):
        physical = Physical()
        model = GatedPosteriorDura(Integrated(), Gate(), physical)
        x = torch.zeros(3, 2, 256)
        out, active = model(x)
        self.assertFalse(active.any())
        self.assertEqual(physical.count, 0)
        torch.testing.assert_close(out, x*0.5, rtol=0, atol=0)

    def test_physical_posterior_uses_weighted_conditional_means(self):
        x = torch.randn(2, 2, 256)
        means = torch.zeros(2, 512)
        covariances = torch.eye(512).repeat(2, 1, 1)
        priors = torch.tensor([0.25, 0.75])

        def posterior(y, action, noise, actual_means, actual_covariances, actual_priors):
            self.assertEqual(y.shape, (2, 512))
            self.assertEqual(noise.shape, (2, 512, 512))
            self.assertTrue(torch.equal(actual_priors, priors))
            states = torch.stack((y, y+4), dim=1)
            return states, priors.log().expand(len(y), -1)

        model = PhysicalPosterior(Blind(), means, covariances, priors,
                                  action_matrix=lambda x: x, posterior=posterior)
        waveform, logvar = model(x)
        torch.testing.assert_close(waveform, x+3)
        torch.testing.assert_close(logvar, torch.zeros(2))
        self.assertFalse(model.blind.training)

    def test_nonlocal_prior_receives_observed_and_conditional_waveforms(self):
        x = torch.randn(2, 2, 256)
        means = torch.zeros(2, 512)
        covariances = torch.eye(512).repeat(2, 1, 1)
        priors = torch.tensor([0.25, 0.75])

        def posterior(y, action, noise, actual_means, actual_covariances, actual_priors):
            return torch.stack((y, y+4), dim=1), priors.log().expand(len(y), -1)

        class Synthesis(nn.Module):
            def forward(self, base, candidates, logvar):
                self_shape = candidates.shape
                if self_shape != (len(base), 3, 2, 256):
                    raise AssertionError('Observed waveform and two conditional waveforms required')
                torch.testing.assert_close(candidates[:, 0], x)
                return base+2

        model = PhysicalPosterior(Blind(), means, covariances, priors,
                                  action_matrix=lambda value: value, posterior=posterior,
                                  synthesizer=Synthesis()).train()
        waveform, _ = model(x)
        torch.testing.assert_close(waveform, x+5)
        self.assertFalse(model.synthesizer.training)


if __name__ == '__main__':
    unittest.main()
