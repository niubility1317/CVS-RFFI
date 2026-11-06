import unittest
import queue
from types import SimpleNamespace

import torch

from experiments.phase1_dura_orthogonal.parallel_source_benchmark import encode, wait_ready, collect_results


class ParallelSourceTest(unittest.TestCase):
    def test_encoding_is_bitwise_and_preserves_shape_and_dtype(self):
        positive = encode([torch.tensor([0.])])
        self.assertNotEqual(positive, encode([torch.tensor([-0.])]))
        self.assertNotEqual(positive, encode([torch.tensor([[0.]])]))
        self.assertNotEqual(positive, encode([torch.tensor([0.], dtype=torch.float64)]))

    def test_failed_worker_is_detected_without_waiting_for_timeout(self):
        with self.assertRaisesRegex(RuntimeError, 'exited before start'):
            wait_ready(queue.Queue(), [SimpleNamespace(exitcode=2)], timeout=30)
        with self.assertRaisesRegex(RuntimeError, 'failed during inference'):
            collect_results(queue.Queue(), [SimpleNamespace(exitcode=-9)], timeout=30)


if __name__ == '__main__':
    unittest.main()
