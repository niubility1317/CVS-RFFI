import unittest

from experiments.phase1_dura_posterior.resources import select_gpu


class ResourceTest(unittest.TestCase):
    def test_respects_existing_processes_and_free_memory(self):
        inventory = [(0, 'gpu-a', 20000), (1, 'gpu-b', 9000), (2, 'gpu-c', 15000)]
        processes = [('gpu-a', 10), ('gpu-a', 11), ('gpu-b', 12), ('gpu-c', 13)]
        self.assertEqual(select_gpu(inventory, processes), 2)
        self.assertIsNone(select_gpu(inventory[:2], processes[:3]))

    def test_invalid_inventory_is_rejected(self):
        with self.assertRaises(ValueError):
            select_gpu([(0, 'gpu-a', 20000), (0, 'gpu-b', 20000)], [])
        with self.assertRaises(ValueError):
            select_gpu([(0, 'gpu-a', 20000)], [('unknown', 1)])


if __name__ == '__main__':
    unittest.main()
