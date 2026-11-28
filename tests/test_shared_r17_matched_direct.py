import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_r17_matched_direct_run as control
from experiments.pamf_lora_main import shared_r17_teacher_direct_run as teacher


class MatchedR17DirectTest(unittest.TestCase):
    def test_seed_and_budget_match(self):
        self.assertEqual((control.SEED, control.STEPS), (teacher.SEED, teacher.STEPS))

    def test_predict_uses_f_free_guard_and_matching_seed(self):
        def check(_rt, _dataset, **kwargs):
            self.assertEqual((control.shared.SEED, control.shared.STEPS), (control.SEED, control.STEPS))
            self.assertFalse(kwargs.get('allow_source_f_guidance', False))
        with patch.object(control, 'read', return_value={}), patch.object(control, 'fit'), \
             patch.object(control.shared, 'predict', side_effect=check), \
             patch.object(control.subprocess, 'run') as score:
            score.return_value.returncode = 0
            control.worker('runtime.json', 'lora')


if __name__ == '__main__':
    unittest.main()
