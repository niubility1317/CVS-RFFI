from copy import deepcopy
import unittest
from experiments.pamf_u_history.fonly_source_analysis import counts,rule,RAW_VIEWS,SEED


class FonlySourceTests(unittest.TestCase):
    def test_G_cells_cannot_change_raw_counts(self):
        h=dict(status='COVARIANCE_SOURCE_EVALUATED',method='CovSel-U',model_seed=SEED,source_V_seen=27000,target_reads=0,U_truth_read=False,
            parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,results=[dict(view=v,n=27000,correct=24000) for v in RAW_VIEWS])
        expected=counts(h,'CovSel-U');other=deepcopy(h);other['results'].append(dict(view='G_raw',n=1,correct=-999))
        self.assertEqual(expected,counts(other,'CovSel-U'))
        other['results'][0]['n']=26999
        with self.assertRaises(ValueError):counts(other,'CovSel-U')

    def test_raw_only_rule_preserves_floor_and_U_budget(self):
        state=dict(calls=6660,seen=1701000,unique_seen=56700,min_visits=30,max_visits=30,covariance_U_seen=1701000,
            U_truth_read=False,target_reads=0,model_EMA_updates=0,new_inference_parameters=0)
        b={v:24000 for v in RAW_VIEWS};a=dict(b,raw=23973)
        self.assertTrue(rule(a,b,state)['source_necessary_pass'])
        a['raw']=23972;self.assertFalse(rule(a,b,state)['source_necessary_pass'])
        a['raw']=24000;a[RAW_VIEWS[1]]-=1
        self.assertFalse(rule(a,b,state)['source_necessary_pass'])
        a=dict(b,G_raw=24000)
        with self.assertRaises(ValueError):rule(a,b,state)
        self.assertFalse(rule(b,b,dict(state,seen=1700000))['source_necessary_pass'])
