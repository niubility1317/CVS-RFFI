import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main.shared_dura_realgain import SharedConvexGain, SharedDuraRealGain, SharedPacketSelector, SharedTwoPriorGain


class IdentityAction:
    def __call__(self, value):
        return value

    def adjoint(self, value):
        return value


class SharedDuraRealGainTest(unittest.TestCase):
    def test_packet_selector_uses_one_expert_and_supports_both_packet_lengths(self):
        torch.manual_seed(17)
        short, long = SharedPacketSelector(32), SharedPacketSelector(1024)
        long.load_state_dict(short.state_dict(), strict=True)
        for model, width in ((short, 32), (long, 1024)):
            y = torch.randn(2, 8 * width, dtype=torch.complex64)
            a, b = torch.randn_like(y), torch.randn_like(y)
            logits = model.logits(y, a, b, IdentityAction())
            self.assertEqual(logits.shape, (2,))
            torch.testing.assert_close(model(y, a, b, IdentityAction()), b, rtol=0, atol=0)
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
            self.assertTrue(torch.isfinite(model.head.weight.grad).all())
            self.assertGreater(model.head.weight.grad.norm().item(), 0)
            with torch.no_grad():
                model.head.bias.fill_(-1.)
            torch.testing.assert_close(model(y, a, b, IdentityAction()), a, rtol=0, atol=0)

    def test_convex_gate_uses_same_weights_and_stays_between_experts(self):
        torch.manual_seed(15)
        short, long = SharedConvexGain(32), SharedConvexGain(1024)
        long.load_state_dict(short.state_dict(), strict=True)
        for model, width in ((short, 32), (long, 1024)):
            y = torch.randn(2, 8 * width, dtype=torch.complex64)
            a, b = torch.randn_like(y), torch.randn_like(y)
            result = model(y, a, b, IdentityAction())
            torch.testing.assert_close(result, model.initial_reference(a, b), rtol=1e-5, atol=2e-6)
            (result-y).abs().square().mean().backward()
            self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
            self.assertGreater(model.net[-1].weight.grad.norm().item(), 0)
            with torch.no_grad():
                model.net[-1].weight.normal_(0, .05)
                model.net[-1].bias.fill_(.2)
            result = model(y, a, b, IdentityAction())
            ratio = (torch.fft.fft((result-a).unflatten(-1, (8, width)), norm='ortho') /
                     torch.fft.fft((b-a).unflatten(-1, (8, width)), norm='ortho'))
            self.assertTrue((ratio.real >= -1e-4).all() and (ratio.real <= 1+1e-4).all())
            self.assertLess(ratio.imag.abs().max().item(), 1e-4)

    def test_lora_r69_adaptation_is_a_matched_control_with_distinct_provenance(self):
        from experiments.pamf_lora_main.shared_dura_realgain_run import screening_status
        comparisons = {'lora_r69_adapt': {'passes': True}}
        available, promote, unavailable = screening_status('lora', comparisons, {'lora_r69_adapt': 0}, 1)
        self.assertTrue(available)
        self.assertTrue(promote)
        self.assertIn('original fixed DURA v1', unavailable[0])

    def test_two_prior_core_shares_parameters_and_preserves_real_gain_at_start(self):
        torch.manual_seed(13)
        short, long = SharedTwoPriorGain(32), SharedTwoPriorGain(1024)
        long.load_state_dict(short.state_dict(), strict=True)
        for model, width in ((short, 32), (long, 1024)):
            received = torch.randn(2, 8 * width, dtype=torch.complex64)
            real_gain = torch.randn_like(received)
            r17 = torch.randn_like(received)
            result = model(received, real_gain, r17, IdentityAction())
            torch.testing.assert_close(result, real_gain, rtol=0, atol=0)
            (result - received).abs().square().mean().backward()
            self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
            self.assertGreater(model.net[-1].weight.grad.norm().item(), 0)

    def test_two_prior_training_matches_packet_normalization(self):
        from experiments.pamf_lora_main.shared_twoprior_run import restore_scaled
        model = SharedTwoPriorGain(32)
        seen = []

        def r17(y):
            seen.append(y.abs().square().mean(-1).sqrt())
            return .8 * y, IdentityAction()

        raw = torch.randn(2, 256, dtype=torch.complex64) * torch.tensor([.2, 7.])[:, None]
        restored, prior = restore_scaled(r17, lambda y: .5*y, model, raw)
        torch.testing.assert_close(seen[0], torch.ones(2), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(restored, .5 * raw, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(restored, prior, rtol=0, atol=0)

    def test_both_packet_lengths_share_parameters_and_initial_r17(self):
        torch.manual_seed(7)
        for width in (32, 1024):
            model = SharedDuraRealGain(width)
            y = torch.randn(2, 8 * width, dtype=torch.complex64)
            prior = torch.randn_like(y)
            actual = model(y, prior, IdentityAction())
            torch.testing.assert_close(actual, prior, rtol=0, atol=0)
        short = SharedDuraRealGain(32)
        long = SharedDuraRealGain(1024)
        long.load_state_dict(short.state_dict(), strict=True)

    def test_real_frequency_gains_preserve_global_phase_and_receive_gradients(self):
        torch.manual_seed(9)
        model = SharedDuraRealGain(32)
        y = torch.randn(3, 256, dtype=torch.complex64)
        prior = torch.randn_like(y)
        output = model(y, prior, IdentityAction())
        (output - y).abs().square().mean().backward()
        self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
        self.assertGreater(model.net[-1].weight.grad.norm().item(), 0)
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .01)
            model.net[-1].bias.fill_(.1)
        rotation = torch.polar(torch.tensor(1.0), torch.tensor(.71))
        rotated = model(rotation * y, rotation * prior, IdentityAction())
        torch.testing.assert_close(rotated, rotation * model(y, prior, IdentityAction()), rtol=1e-5, atol=2e-6)

    def test_rejects_unregistered_or_mismatched_packet_shapes(self):
        with self.assertRaises(ValueError):
            SharedDuraRealGain(64)
        with self.assertRaises(ValueError):
            SharedDuraRealGain(32)(torch.ones(1, 256), torch.ones(1, 256), IdentityAction())
        with self.assertRaises(ValueError):
            SharedDuraRealGain(32)(torch.ones(1, 256, dtype=torch.complex64),
                                   torch.ones(1, 128, dtype=torch.complex64), IdentityAction())

    def test_scorer_rejects_gpu_and_unfrozen_predictions(self):
        from experiments.pamf_lora_main.shared_dura_realgain_run import score
        runtime = {'run_root': 'unused', 'project': 'unused'}
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'Independent CPU scorer'):
                score(runtime, 'wisig')
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), patch(
                'experiments.pamf_lora_main.shared_dura_realgain_run.read',
                return_value=dict(status='RUNNING', views=['clean', 'practical_low_urban'],
                                  modes=['direct', 'wisig_r17', 'real_gain', 'dura_v1', 'fusion'],
                                  count=360, source_truth_reads=0, target_reads=0)):
            with self.assertRaisesRegex(ValueError, 'must be frozen before truth'):
                score(runtime, 'wisig')

    def test_training_and_inference_use_same_per_packet_normalization(self):
        from experiments.pamf_lora_main.shared_dura_realgain_run import restore_scaled
        model = SharedDuraRealGain(32)
        seen = []

        def base(normalized):
            seen.append(normalized.abs().square().mean(-1).sqrt())
            return normalized * .5, IdentityAction()

        raw = torch.randn(2, 256, dtype=torch.complex64) * torch.tensor([.2, 7.])[:, None]
        result, prior = restore_scaled(base, model, raw)
        torch.testing.assert_close(seen[0], torch.ones(2), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result, .5 * raw, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result, prior, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
