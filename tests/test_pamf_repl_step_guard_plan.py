import unittest
from experiments.pamf_repl_step_guard import plan


class PlanTests(unittest.TestCase):
    def test_new_run_identity_passes_before_missing_input(self):
        for row in plan.candidate_rows():
            with self.assertRaises(FileNotFoundError):plan.configuration({},row,'/missing','commit')
            self.assertEqual(row['recipe'],'step_guard')
            self.assertEqual(row['views'],['clean'])
    def test_reject_unknown_row(self):
        row=dict(plan.candidate_rows()[0],recipe='other')
        with self.assertRaises(ValueError):plan.configuration({},row,'/missing','commit')


if __name__=='__main__':unittest.main()
