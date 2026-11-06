import unittest

import torch

from experiments.phase1_dura_orthogonal.parallel_source_benchmark import encode


class ParallelSourceTest(unittest.TestCase):
    def test_encoding_is_bitwise_and_preserves_shape_and_dtype(self):
        positive = encode([torch.tensor([0.])])
        self.assertNotEqual(positive, encode([torch.tensor([-0.])]))
        self.assertNotEqual(positive, encode([torch.tensor([[0.]])]))
        self.assertNotEqual(positive, encode([torch.tensor([0.], dtype=torch.float64)]))


if __name__ == '__main__':
    unittest.main()
