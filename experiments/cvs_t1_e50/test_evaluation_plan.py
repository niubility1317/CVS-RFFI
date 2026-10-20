import importlib.util
from pathlib import Path
import unittest

s=importlib.util.spec_from_file_location('evaluation_plan',Path(__file__).with_name('evaluation_plan.py'))
plan=importlib.util.module_from_spec(s);s.loader.exec_module(plan)


class TestTruthLast(unittest.TestCase):
    def header(self):
        return dict(status='PREDICTIONS_COMPLETE',views=list(plan.VIEWS),truth_read=False,
                    query_fit=False,all_parameters_frozen=True,head_fit_updates=0)

    def test_complete_frozen_seven_views(self):
        plan.require_seven_views(self.header())

    def test_partial_views_and_premature_truth_rejected(self):
        for key,value in [('views',['clean']),('views',list(plan.VIEWS[:-1])),('truth_read',True),
                          ('query_fit',True),('all_parameters_frozen',False),('head_fit_updates',1)]:
            h=self.header();h[key]=value
            with self.assertRaises(ValueError):plan.require_seven_views(h)


if __name__=='__main__':unittest.main()
