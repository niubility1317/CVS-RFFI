import unittest
from unittest.mock import patch

import torch

from experiments.phase1_dura_orthogonal.shared_source_benchmark import independent, shared, compare, same_bits


class Reader:
    def __init__(self):
        self.calls = 0

    def features(self, x):
        self.calls += 1
        return x*2

    def classify_features(self, x):
        return x.sum(-1)


class SharedBenchmarkTest(unittest.TestCase):
    def test_signed_zero_is_not_bitwise_equal(self):
        self.assertFalse(same_bits(torch.tensor([0.]), torch.tensor([-0.])))

    def test_exact_outputs_and_shared_call_count(self):
        reader = Reader()
        models = [lambda x, a, d, v, n=n: a+n*d for n in (1, 2, 3)]
        triples = [(model, object(), reader) for model in models]
        x = torch.tensor([[1., 2., 3.]])
        def context(dura, value):
            return value+1, value*.25, torch.ones(len(value))
        with patch('experiments.phase1_dura_orthogonal.shared_source_benchmark.extract_context', context) as mocked:
            original = independent(x, triples)
            self.assertEqual(reader.calls, 9)
            reader.calls = 0
            common = shared(x, triples)
            self.assertEqual(reader.calls, 5)
            compare(original, common)


if __name__ == '__main__':
    unittest.main()
