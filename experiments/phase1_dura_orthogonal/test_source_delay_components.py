import unittest
import torch

from experiments.pamf_action_restore.operators import Action
from experiments.phase1_dura_orthogonal.source_delay_components import tap_windows, component_metrics


class DelayComponentTest(unittest.TestCase):
    def test_tap_partition_preserves_widely_linear_action(self):
        torch.manual_seed(19)
        h = torch.complex(torch.randn(2, 64), torch.randn(2, 64))
        main = torch.complex(torch.randn(2, 256), torch.randn(2, 256))
        image = torch.complex(torch.randn(2, 256), torch.randn(2, 256))
        x = torch.complex(torch.randn(2, 256), torch.randn(2, 256))
        windows = tap_windows(h)
        torch.testing.assert_close(windows.sum(1), h)
        parts = torch.stack([Action(v, main, image)(x) for v in windows.unbind(1)], 1)
        torch.testing.assert_close(parts.sum(1), Action(h, main, image)(x), rtol=2e-5, atol=1e-5)

    def test_metrics_and_invalid_inputs(self):
        components = torch.zeros(2, 4, 256, dtype=torch.complex64)
        components[:, 2] = 1+1j
        ratios, coherence, residual, dominant = component_metrics(components, components.sum(1))
        torch.testing.assert_close(ratios[:, 2], torch.ones(2))
        torch.testing.assert_close(ratios[:, :2], torch.zeros(2, 2))
        torch.testing.assert_close(coherence, torch.zeros(2, 6))
        torch.testing.assert_close(residual, torch.zeros(2))
        self.assertEqual(dominant.tolist(), [2, 2])
        with self.assertRaises(ValueError):
            tap_windows(torch.zeros(1, 63, dtype=torch.complex64))
        with self.assertRaises(ValueError):
            component_metrics(components, torch.ones(1, 256, dtype=torch.complex64))


if __name__ == '__main__':
    unittest.main()
