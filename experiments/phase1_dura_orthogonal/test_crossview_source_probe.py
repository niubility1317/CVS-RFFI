import unittest

import torch

from experiments.phase1_dura_orthogonal.crossview_source_probe import decompose


class CrossViewSourceTest(unittest.TestCase):
    def test_common_error_and_zero_mean_variation(self):
        clean = torch.ones(2, 2, 256)
        outputs = clean[:, None].repeat(1, 3, 1, 1)
        outputs[0] += 2
        outputs[1, 0] += 1
        outputs[1, 1] -= 1
        result = decompose(outputs, clean)
        torch.testing.assert_close(result[0], torch.tensor([4., 4., 0.]))
        torch.testing.assert_close(result[1], torch.tensor([2/3, 0., 2/3]))


if __name__ == '__main__':
    unittest.main()
