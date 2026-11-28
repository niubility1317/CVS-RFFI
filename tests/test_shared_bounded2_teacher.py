import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_bounded2_teacher_run as run


class BoundedTwoPercentTest(unittest.TestCase):
    def test_worker_reconstructs_the_registered_cap_and_anchor(self):
        def check(_rt, dataset, **kwargs):
            model = run.shared.SharedTwoPriorGain(32)
            self.assertEqual(model.correction_cap, .02)
            self.assertEqual(bool(model.anchor_real), run.ANCHOR_REAL[dataset])
            self.assertTrue(kwargs.get('allow_source_f_guidance'))
        with patch.object(run, 'read', return_value={}), patch.object(run, 'fit'), \
             patch.object(run.shared, 'predict', side_effect=check), \
             patch.object(run.subprocess, 'run') as score:
            score.return_value.returncode = 0
            run.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
