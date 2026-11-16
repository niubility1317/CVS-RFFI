import runpy
import unittest
from copy import deepcopy
from pathlib import Path

from experiments.pamf_tri_label_only import plan


class LabeledOnlyControls(unittest.TestCase):
    def setUp(self):
        self.ctx=runpy.run_path(str(Path(__file__).with_name('test_pamf_paper_completion.py')))['context']()

    def test_fixed_scratch_rows_and_no_U(self):
        rows=plan.candidate_rows()
        self.assertEqual(len(rows),2)
        for row in rows:
            c=plan.configuration(self.ctx,row,'/project','commit')
            self.assertEqual(c['ablation'],'F20')
            self.assertFalse(c['U_use'])
            self.assertEqual(c['regime'],'matched_sat_supervised')
            self.assertEqual(c['trajectory'],[20,10,20])
            self.assertTrue(row['permuted'])
            self.assertEqual(row['views'],['clean'])
            self.assertFalse(any(k.startswith('hard_pl') for k in c))
            self.assertNotIn('parent_source',c)
            plan.validate(c)

    def test_rejects_truth_and_unlabeled_settings(self):
        c=plan.configuration(self.ctx,plan.candidate_rows()[0],'/project','commit')
        for changes in ({'truth':'forbidden'},{'U_use':True},{'regime':'matched_sat_plain_pl'},
                        {'checkpoint':'forbidden'}):
            with self.assertRaises(ValueError):plan.validate(dict(c,**changes))
        bad=deepcopy(c);bad['paper_row']['permuted']=False
        with self.assertRaises(ValueError):plan.validate(bad)


if __name__=='__main__':unittest.main()
