import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_bounded2_matched_run as control
from experiments.pamf_lora_main import shared_bounded2_teacher_run as teacher


class MatchedBoundedTwoPercentTest(unittest.TestCase):
    def test_seed_budget_and_source_anchors_match(self):
        self.assertEqual((control.SEED, control.STEPS, control.ANCHOR_REAL),
                         (teacher.SEED, teacher.STEPS, teacher.ANCHOR_REAL))

    def test_prediction_reconstructs_two_percent_and_f_free_guard(self):
        def check(_rt, dataset, **kwargs):
            model = control.shared.SharedTwoPriorGain(32)
            self.assertEqual(model.correction_cap, .02)
            self.assertEqual(bool(model.anchor_real), control.ANCHOR_REAL[dataset])
            self.assertFalse(kwargs.get('allow_source_f_guidance', False))
        with patch.object(control, 'read', return_value={}), patch.object(control, 'fit'), \
             patch.object(control.shared, 'predict', side_effect=check), \
             patch.object(control.subprocess, 'run') as score:
            score.return_value.returncode = 0
            control.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
