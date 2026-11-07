import unittest
import torch
from experiments.pamf_phase3_restoration.waveform_mixture_prior import mixture_posterior
from experiments.pamf_phase3_restoration.mixture_gate_posterior import mixture_components, combine_components


class MixtureGatePosteriorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.action = torch.randn(3, 6, 6, dtype=torch.float64)
        self.observed = torch.randn(3, 6, dtype=torch.float64)
        self.mean = torch.randn(6, dtype=torch.float64)
        x = torch.randn(6, 6, dtype=torch.float64)
        self.within = x @ x.T + .3*torch.eye(6, dtype=torch.float64)
        self.offsets = torch.randn(4, 6, dtype=torch.float64)
        self.weights = torch.tensor([.2, .3, .1, .4], dtype=torch.float64)
        self.variance = torch.tensor([.1, .4, .7], dtype=torch.float64)

    def test_zero_adjustment_matches_exact_existing_mixture(self):
        components, logits, residual = mixture_components(self.action, self.observed, self.mean,
            self.within, self.offsets, self.weights, self.variance)
        actual, probs = combine_components(components, logits, torch.zeros_like(logits))
        reference, original_probs = mixture_posterior(self.action, self.observed, self.mean,
            self.within, self.offsets, self.weights, self.variance)
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(probs, original_probs, rtol=1e-10, atol=1e-10)
        self.assertLess(residual.max().item(), 1e-8)

    def test_gate_gets_finite_gradient_and_probabilities_sum_one(self):
        components, logits, _ = mixture_components(self.action, self.observed, self.mean,
            self.within, self.offsets, self.weights, self.variance)
        adjust = torch.zeros_like(logits, requires_grad=True)
        actual, probs = combine_components(components, logits, adjust)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(adjust.grad).all())
        self.assertGreater(adjust.grad.abs().sum().item(), 0)
        torch.testing.assert_close(probs.sum(1), torch.ones(3, dtype=torch.float64))

    def test_invalid_prior_rejected(self):
        with self.assertRaises(ValueError):
            mixture_components(self.action, self.observed, self.mean, self.within,
                               self.offsets, torch.tensor([.2, .3, -.1, .6]), self.variance)


class FullRestorerTest(unittest.TestCase):
    def test_initial_mix_and_branch_gradient_isolation(self):
        from experiments.pamf_phase3_restoration.mixture_gate_model import scratch_mixture_gate
        torch.set_num_threads(2)
        model = scratch_mixture_gate(7)
        offsets = torch.zeros(4, 512)
        offsets[0, 0], offsets[1, 0] = .4, -.4
        offsets[2, 1], offsets[3, 1] = .4, -.4
        weights = torch.full((4,), .25)
        covariance = torch.eye(512)
        within = covariance - (offsets.T * weights) @ offsets
        model.physical.bind_prior(torch.zeros(512), covariance)
        model.physical.bind_mixture(within, offsets, weights)
        received = torch.randn(2, 2, 256) * .3
        restored, aux = model(received, return_aux=True)
        torch.testing.assert_close(restored, aux['base_iq'], rtol=0, atol=0)
        self.assertLess(aux['relative_solve_residual'].max().item(), 1e-8)
        restored.square().mean().backward()
        self.assertTrue(torch.isfinite(model.detail.output.weight.grad).all())
        self.assertGreater(model.detail.output.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.detail.refiner.output.weight.grad).all())
        self.assertGreater(model.detail.refiner.output.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.physical.parameters()))


class SourceContractTest(unittest.TestCase):
    def test_prior_does_not_require_numpy_torch_bridge(self):
        import numpy as np
        from unittest.mock import patch
        from experiments.pamf_phase3_restoration.mixture_gate_trial import prior_tensor
        a = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
        with patch('torch.from_numpy', side_effect=RuntimeError('bridge unavailable')):
            actual = prior_tensor(a)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(actual.tolist(), a.tolist())

    def test_mixture_prior_rejects_forbidden_source(self):
        from experiments.pamf_phase3_restoration.mixture_gate_trial import validate_mixture_metadata, PRIOR_COMMIT
        result = dict(status='SOURCE_WAVEFORM_MIXTURE_COMPLETE', code_commit=PRIOR_COMMIT,
                      TX_labels_read=0, OE_reads=0, novel_reads=0, target_inputs=0,
                      optimizer_updates=0, source_prior={'source_Ls':6300, 'source_Us':56700})
        complete = {'status':'ARTIFACTS_COMPLETE'}
        validate_mixture_metadata(result, complete)
        for key in ('TX_labels_read', 'OE_reads', 'novel_reads', 'target_inputs', 'optimizer_updates'):
            changed = dict(result, **{key:1})
            with self.assertRaises(ValueError):
                validate_mixture_metadata(changed, complete)
        with self.assertRaises(ValueError):
            validate_mixture_metadata(dict(result, source_prior={'source_Ls':6300,'source_Us':1}), complete)


if __name__ == '__main__':
    unittest.main()
