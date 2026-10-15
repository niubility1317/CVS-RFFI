"""Protocol negatives for the frozen source probe; no real IQ or checkpoint."""
import unittest
import torch
from .probe import gate, summary, validate_source
from .probe_design import config


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.V=dict(x=torch.zeros(27000,2,256), y=torch.arange(27000)%6,
                   rx=torch.tensor([1,3,4,6,8]).repeat(5400), day=torch.arange(27000)%3+1,
                   ids=['V:'+str(i) for i in range(27000)])
        cls.contract=dict(role_ids=dict(V=cls.V['ids'],L_s=['L:0'],U_s=['U:0']))

    def test_source_contract_and_illegal_roles(self):
        validate_source(self.V,self.contract)
        for change in (dict(ids=self.V['ids'][:-1]),dict(y=torch.full((27000,),-1)),
                       dict(rx=torch.full((27000,),11)),dict(day=torch.zeros(27000)),
                       dict(ids=['L:0']+self.V['ids'][1:])):
            with self.assertRaises(ValueError):
                validate_source(dict(self.V,**change),self.contract)

    def test_threshold_does_not_use_target_results(self):
        values={k:dict(overall=dict(accuracy=v)) for k,v in
                [('RAW',.95),('G_E1_REFLECT',.945),('G_E20_REFLECT',.94)]}
        self.assertEqual(gate(values)['status'],'PASS')
        values['G_E20_REFLECT']['overall']['accuracy']=.939
        self.assertEqual(gate(values)['status'],'FAIL')
        self.assertFalse(gate(values)['target_metrics_read'])

    def test_repairs_and_fixed_correct_set(self):
        y=torch.arange(6);base=torch.tensor([0,1,2,3,0,0]);pred=torch.tensor([0,1,2,0,4,5])
        m=summary(pred,y,torch.ones(6),base)
        self.assertEqual(m['reference_clean_correct'],4)
        self.assertEqual(m['repairs'],2)
        self.assertEqual(m['new_errors'],1)
        self.assertEqual(m['clean_correct_retention'],.75)

    def test_scope_is_zero_update_full_source_V(self):
        c=config()
        self.assertEqual(c['forward_packets'],135002)
        self.assertEqual(c['source_count'],27000)
        self.assertEqual(c['parent_training_epochs'],20)
        self.assertTrue(c['source_only'])
        for field in ('query_read','optimizer_updates','EMA_updates','head_fit_updates','U_opened','target_predictions'):
            self.assertFalse(c[field])


if __name__=='__main__':
    unittest.main()
