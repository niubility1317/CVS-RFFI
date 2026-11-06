import unittest

import torch

from experiments.phase1_dura_orthogonal.direct_source_probe import pair_errors


class DirectSourceProbeTest(unittest.TestCase):
    def test_order_and_per_record_normalization(self):
        clean = torch.ones(2, 2, 256)
        values = pair_errors(clean*2, clean*1.5, clean, clean)
        torch.testing.assert_close(values, torch.tensor([[1., .25, 0.], [1., .25, 0.]]))


if __name__ == '__main__':
    unittest.main()
