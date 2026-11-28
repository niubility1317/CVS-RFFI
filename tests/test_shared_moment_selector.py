import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main.shared_moment_selector_run import moment_features
from experiments.pamf_lora_main import shared_moment_selector_run as runner


class SharedMomentSelectorTest(unittest.TestCase):
    def test_features_are_finite_and_invariant_to_complex_gain(self):
        torch.manual_seed(19)
        x = torch.randn(4, 256, dtype=torch.complex64)
        gain = torch.tensor([1 + 2j, -.7 + .2j, 3 - 1j, .3j])[:, None]
        torch.testing.assert_close(moment_features(x), moment_features(x * gain), rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.isfinite(moment_features(torch.zeros_like(x))).all())

    def test_features_detect_temporal_distortion(self):
        torch.manual_seed(29)
        x = torch.randn(4, 256, dtype=torch.complex64)
        filtered = x + .8 * x.roll(1, dims=-1)
        delta = (moment_features(x) - moment_features(filtered)).abs().sum(-1)
        self.assertTrue((delta > .05).all())

    def test_rejects_non_complex_or_short_inputs(self):
        with self.assertRaises(ValueError):
            moment_features(torch.ones(4, 256))
        with self.assertRaises(ValueError):
            moment_features(torch.ones(4, 8, dtype=torch.complex64))

    def test_worker_predicts_with_the_actual_training_budget(self):
        def check_budget(_runtime, _dataset):
            self.assertEqual(runner.shared.STEPS, runner.STEPS)

        with patch.object(runner, 'read', return_value={}), patch.object(runner, 'fit'), \
             patch.object(runner.shared, 'predict', side_effect=check_budget), \
             patch.object(runner.subprocess, 'run') as score:
            score.return_value.returncode = 0
            runner.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
