import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('t1_e50_plan', Path(__file__).with_name('plan.py'))
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class BudgetTests(unittest.TestCase):
    def test_authorized_two_paths(self):
        scratch = plan.profile('SCRATCH50')
        tail = plan.profile('E30PLUS20')
        self.assertEqual((scratch['epochs'], scratch['label_epochs'], scratch['pseudo_epochs']), (50, 33, 17))
        self.assertEqual((tail['epochs'], tail['label_epochs'], tail['pseudo_epochs']), (20, 0, 20))
        self.assertEqual(scratch['updates'], 11100)
        self.assertEqual(tail['updates'], 4440)
        self.assertEqual(tail['step_offset'] + tail['updates'], 11100)
        self.assertEqual(scratch['model_seed'], 2026092701)
        self.assertEqual(tail['model_seed'], 2026092701)

    def test_unregistered_arm_rejected(self):
        for arm in ('T6', 'KD', 'SCRATCH20', '', None):
            with self.assertRaises(ValueError):
                plan.profile(arm)

    def test_channel_clock_crosses_E30_and_reaches_E50(self):
        self.assertEqual(plan.training_clock('E30PLUS20', 1, 1), (31, 6661))
        self.assertEqual(plan.training_clock('E30PLUS20', 20, 4440), (50, 11100))
        self.assertEqual(plan.training_clock('SCRATCH50', 50, 11100), (50, 11100))
        for arm, epoch, step in [('E30PLUS20', 21, 4441), ('SCRATCH50', 51, 11101), ('SCRATCH50', 1, 223)]:
            with self.assertRaises(ValueError):
                plan.training_clock(arm, epoch, step)

    def test_budget_and_truth_access_mutations_rejected(self):
        for arm in plan.ARMS:
            for key, value in [('epochs', 51), ('model_seed', 2026092702), ('As_KL', 1.), ('target_training', True), ('label_epochs', 1)]:
                changed = plan.profile(arm)
                changed[key] = value
                with self.assertRaises(ValueError):
                    plan.validate(changed)


if __name__ == '__main__':
    unittest.main()
