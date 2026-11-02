import unittest
from experiments.phase1_adaptive_hankel import ground_controller, source_controller
from experiments.phase1_dura_posterior import source_controller as dura_controller

class ResourceTest(unittest.TestCase):
    def test_source_and_ground_share_dura_lease(self):
        self.assertEqual(ground_controller.GPU_LOCK, dura_controller.GPU_LOCK)
        self.assertEqual(source_controller.GPU_LOCK, dura_controller.GPU_LOCK)

if __name__ == '__main__': unittest.main()
