import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from . import policy_audit as m, actions as a


def fixture():
    g=torch.Generator().manual_seed(72)
    x=torch.randn(12,2,256,generator=g)
    cache=dict(x=x,y=torch.arange(12)%3,state=a.old.signal_state(x),ids=[str(i) for i in range(12)],
               role='L_s',rx=torch.arange(12)%2,day=torch.arange(12)%2)
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(.2),torch.nn.Linear(512,3))
    return model,cache


class Checks(unittest.TestCase):
    def test_risk_focus_and_detached_reference(self):
        clean=torch.tensor([12.,12.],requires_grad=True);changed=torch.tensor([10.,-2.],requires_grad=True)
        m.search_loss(clean,changed,'risk').backward()
        self.assertIsNone(clean.grad);self.assertEqual(float(changed.grad[0]),0.)
        self.assertGreater(float(changed.grad[1]),0.)

    def test_raw_and_risk_differ(self):
        clean=torch.tensor([12.,12.]);changed=torch.tensor([10.,-2.],requires_grad=True)
        m.search_loss(clean,changed,'raw').backward()
        self.assertTrue(torch.equal(changed.grad,torch.full((2,),.5)))

    def test_role_rejection(self):
        model,c=fixture();c['role']='V'
        with self.assertRaises(ValueError):m.fit(model,c,torch.zeros(12),10,'risk')

    def test_state_preservation_budget_and_private_rng(self):
        model,c=fixture();model.train();before={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone();u=torch.ones(12)
        nets,report=m.fit(model,c,u,10,'risk')
        self.assertTrue(all(report['identity_state'].values()));self.assertEqual(report['auxiliary_updates'],4)
        self.assertEqual(report['sample_occurrences'],48);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(model.training);self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(not p.requires_grad for n in nets.values() for p in n.parameters()))

    def test_equal_draws_across_search_objectives(self):
        model,c=fixture();records=[];original=a._proposal
        def record(*args,**kwargs):
            p=original(*args,**kwargs);records.append((p['radius'].clone(),p['conditional_mask'].clone()));return p
        with patch.object(a,'_proposal',side_effect=record):
            m.fit(model,c,torch.ones(12),91,'raw');m.fit(model,c,torch.ones(12),91,'risk')
        for one,two in zip(records[:4],records[4:]):
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(one,two)))

    def test_eval_pairing_complete_and_no_fit(self):
        model,c=fixture();model.eval();nets,_=m.fit(model,c,torch.ones(12),45,'risk')
        banks={k:nets for k in ('old','fresh_raw','fresh_risk')}
        with tempfile.TemporaryDirectory() as tmp,patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('V optimizer')):
            c['role']='V';result=m.evaluate(model,c,m.clean_forward(model,c),banks,45,'V',Path(tmp)/'eval.pt')
            self.assertEqual(len(result['metrics']),8);self.assertEqual(len(result['paired']),8)
            for r in result['metrics']:self.assertEqual(r['all']['count'],12);self.assertLess(r['radius_max_error'],1e-6)
            for r in result['paired']:
                if r['control']!='coverage':self.assertEqual(r['mean_risk_difference'],0.)

    def test_no_alternate_selection(self):
        rows=[dict(role='V',metrics=[dict(method=method,all=dict(finite_risk=value)) for kind in a.KINDS
              for method,value in zip(m.METHODS,[1.,2.,5.,4.])])]
        result=m.advancement(rows);self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['risk_beats_fresh_raw'])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
