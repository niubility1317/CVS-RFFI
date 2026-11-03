import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.phase1_shared_physical_predict.source_benchmark import tensor_to_reference_array


class TensorBridgeTest(unittest.TestCase):
    def test_uses_numpy_instance_owned_by_comparison_module(self):
        value = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        with patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('torch numpy bridge is incompatible')):
            actual = tensor_to_reference_array(value)
        self.assertIs(type(actual), np.ndarray)
        np.testing.assert_array_equal(actual, np.array([[1, 2], [3, 4]]))
        value[0, 0] = 9
        self.assertEqual(actual[0, 0], 1)


if __name__ == '__main__':
    unittest.main()
