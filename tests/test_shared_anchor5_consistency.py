import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_anchor5_consistency_run as run


class SharedAnchorFivePercentTest(unittest.TestCase):
    def test_worker_reconstructs_registered_cap_and_anchor_for_each_dataset(self):
        def check(_rt, dataset, **kwargs):
            model = run.shared.SharedTwoPriorGain(32)
            self.assertEqual(model.correction_cap, .05)
            self.assertEqual(bool(model.anchor_real), run.ANCHOR_REAL[dataset])
            self.assertTrue(kwargs['allow_source_f_guidance'])
        with patch.object(run, 'read', return_value={}), patch.object(run, 'fit'), \
             patch.object(run.shared, 'predict', side_effect=check), \
             patch.object(run.subprocess, 'run') as score:
            score.return_value.returncode = 0
            for dataset in ('lora', 'wisig'):
                run.worker('runtime.json', dataset)


if __name__ == '__main__':
    unittest.main()
