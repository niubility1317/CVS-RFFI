import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.phase1_cumulant_rank import source_pair_probe as probe

class SourcePairTest(unittest.TestCase):
    def test_wrong_source_checkpoint_rejected(self):
        with patch.object(probe.core,'read',return_value={'status':'SOURCE_TRAINED','schema':'other'}), \
             patch.object(probe.torch,'load',return_value={}):
            with self.assertRaises(ValueError):
                probe.load_fixed(Path('.'),'source',probe.BASE_SCHEMA,'wrong')

if __name__=='__main__':unittest.main()
