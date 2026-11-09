import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.phase1_cumulant_direction_cycle import ground

class GroundContractTest(unittest.TestCase):
    def test_mismatched_checkpoint_is_rejected_before_loading_recognizer(self):
        with patch.object(ground.core, 'read', return_value={'status':'SOURCE_TRAINED','schema':'other'}), \
             patch.object(ground.torch, 'load', return_value={}), \
             patch.object(ground.core, 'teacher', side_effect=AssertionError('recognizer accessed')):
            with self.assertRaises(ValueError):
                ground.load_fixed(Path('.'), 'source', 'wrong', 'cpu')

if __name__ == '__main__': unittest.main()
