import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_matched_direct_run as control
from experiments.pamf_lora_main import shared_teacher_direct_run as teacher


class MatchedDirectTest(unittest.TestCase):
    def test_seed_and_budget_match(self):
        self.assertEqual((control.SEED, control.STEPS), (teacher.SEED, teacher.STEPS))

    def test_worker_predicts_with_matched_budget_and_default_f_free_guard(self):
        def check_predict(_rt, _dataset, **kwargs):
            self.assertEqual((control.shared.SEED, control.shared.STEPS), (control.SEED, control.STEPS))
            self.assertFalse(kwargs.get('allow_source_f_guidance', False))

        with patch.object(control, 'read', return_value={}), patch.object(control, 'fit'), \
             patch.object(control.shared, 'predict', side_effect=check_predict), \
             patch.object(control.subprocess, 'run') as score:
            score.return_value.returncode = 0
            control.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
