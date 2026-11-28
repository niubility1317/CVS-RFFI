import unittest
from unittest.mock import patch

import torch

from experiments.pamf_lora_main import shared_r17_scale_diagnostic as diagnostic
from experiments.pamf_lora_main.shared_r17_scale_diagnostic import NormalizedR17


class R17ScaleDiagnosticTest(unittest.TestCase):
    def test_diagnostic_returns_the_actual_normalized_r17_without_parameters(self):
        model = NormalizedR17(32)
        y = torch.randn(2, 256, dtype=torch.complex64)
        real = torch.randn_like(y)
        r17 = torch.randn_like(y)
        torch.testing.assert_close(model(y, real, r17, None), r17, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)

    def test_independent_scoring_labels_the_actual_diagnostic(self):
        with patch.object(diagnostic, 'score') as score:
            diagnostic.diagnostic_score({}, 'wisig')
            score.assert_called_once_with({}, 'wisig', candidate_name='normalized_r17')


if __name__ == '__main__':
    unittest.main()
