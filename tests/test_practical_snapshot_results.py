import copy
import unittest
from experiments.practical_full_2x2.snapshot_results import validate_panel, VARIANTS, SEEDS


class CompletePanelTests(unittest.TestCase):
    def panel(self):
        return [dict(row=dict(family=f, cell=c, model_seed=s,
                            G_roles='LU' if c in ('B','D','native_T1') else 'L'))
                for f,c in VARIANTS for s in SEEDS]

    def test_complete_registered_variants(self):
        validate_panel(self.panel())

    def test_missing_or_duplicate_seed_cannot_be_summarized(self):
        rows=self.panel()
        with self.assertRaises(ValueError):validate_panel(rows[:-1])
        rows[-1]=copy.deepcopy(rows[-2])
        with self.assertRaises(ValueError):validate_panel(rows)

    def test_supervised_plus_unlabeled_G_is_not_L_only(self):
        rows=self.panel(); rows[0]['row']['G_roles']='LU'
        with self.assertRaisesRegex(ValueError,'G role'):validate_panel(rows)


if __name__=='__main__':unittest.main()
