import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_bounded_matched_run as control
from experiments.pamf_lora_main import shared_bounded_teacher_run as teacher


class MatchedBoundedTest(unittest.TestCase):
    def test_seed_budget_and_anchor_buffers_match_teacher(self):
        self.assertEqual((control.SEED, control.STEPS, control.ANCHOR_REAL),
                         (teacher.SEED, teacher.STEPS, teacher.ANCHOR_REAL))

    def test_prediction_keeps_f_free_guard(self):
        def check(_rt, _dataset, **kwargs):
            self.assertEqual((control.shared.SEED, control.shared.STEPS), (control.SEED, control.STEPS))
            self.assertFalse(kwargs.get('allow_source_f_guidance', False))
        with patch.object(control, 'read', return_value={}), patch.object(control, 'fit'), \
             patch.object(control.shared, 'predict', side_effect=check), \
             patch.object(control.subprocess, 'run') as score:
            score.return_value.returncode = 0
            control.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
