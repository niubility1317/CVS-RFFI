import unittest
from unittest.mock import patch

from experiments.pamf_lora_main import shared_matched_mse_selector_run as control
from experiments.pamf_lora_main import shared_teacher_selector_run as teacher


class MatchedControlTest(unittest.TestCase):
    def test_seed_and_update_budget_match_teacher(self):
        self.assertEqual((control.SEED, control.STEPS), (teacher.SEED, teacher.STEPS))

    def test_worker_uses_same_budget_for_prediction(self):
        def check_budget(_runtime, _dataset):
            self.assertEqual(control.shared.STEPS, control.STEPS)

        with patch.object(control, 'read', return_value={}), patch.object(control, 'fit'), \
             patch.object(control.shared, 'predict', side_effect=check_budget), \
             patch.object(control.subprocess, 'run') as score:
            score.return_value.returncode = 0
            control.worker('runtime.json', 'wisig')


if __name__ == '__main__':
    unittest.main()
