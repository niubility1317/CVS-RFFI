import unittest
from experiments.phase1_cumulant_diverse.train import SCHEDULE, schedule_contract

class ScheduleTest(unittest.TestCase):
    def test_fixed_budget_and_unique_view_coverage(self):
        self.assertEqual(len(schedule_contract()), 20)
        self.assertEqual(SCHEDULE[:10], SCHEDULE[10:])
        self.assertEqual(sorted(v for g in SCHEDULE for v in g), sorted(list(range(1,31))*2))
        self.assertEqual(len(set(SCHEDULE)), 10)

    def test_missing_or_duplicated_views_rejected(self):
        bad = list(SCHEDULE)
        bad[0] = (1, 2, 2)
        with self.assertRaises(ValueError): schedule_contract(bad)
        with self.assertRaises(ValueError): schedule_contract(SCHEDULE[:-1])

if __name__ == '__main__': unittest.main()
