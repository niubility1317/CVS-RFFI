import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.phase1_shared_physical_predict.parallel_benchmark import flatten_batches


class FlattenBatchesTest(unittest.TestCase):
    def test_preserves_boolean_gates_and_prediction_fields(self):
        anchor = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        output = anchor + 1
        active = torch.tensor([True, False])
        classes = [torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])]
        with patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('incompatible torch bridge')):
            actual = flatten_batches([(anchor, output, active, classes)])
        self.assertEqual(actual[2].dtype, np.bool_)
        np.testing.assert_array_equal(actual[2], [True, False])
        np.testing.assert_array_equal(actual[0], [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        np.testing.assert_array_equal(actual[5], [4, 5])


if __name__ == '__main__':
    unittest.main()
