import unittest
from copy import deepcopy
import numpy as np
from . import protocol as p


class ProtocolTests(unittest.TestCase):
    def test_pair_identity_and_union(self):
        q=p.paired([0,1,0,1],[1,1,1,0],[0,1,1,0])
        self.assertEqual((q['repair'],q['damage'],q['both_wrong'],q['both_correct']), (1,2,0,1))
        self.assertEqual(q['delta_pp'], -25)
        self.assertEqual(q['oracle_union_pct'],100)

    def test_aligned_only(self):
        for args in [([],[],[]),([0],[0,1],[0]),([[0]],[[0]],[[0]])]:
            with self.assertRaises(ValueError): p.paired(*args)

    def test_both_draws_required(self):
        rows={(d,s):{'delta_pp':1.} for d in (0,1) for s in p.SCENES}
        self.assertTrue(p.qualification(rows)['pass_both_draws'])
        rows[(1,p.SCENES[2])]['delta_pp']=.99
        self.assertFalse(p.qualification(rows)['pass_both_draws'])

    def test_provenance_negative(self):
        c={'role_ids':{'L_s':['l'],'U_s':['u'],'V':['v']},'classes':list(p.CLASSES)}
        init=dict(status='SCRATCH',checkpoint=None,ancestors=[],checkpoint_sources=[],scratch_only=True,
                  model_seed=2026092701,selection='fixed_last_epoch',target_contact=False,target_access=False)
        done=dict(status='SOURCE_TRAINED',epoch=200,steps=9800,target_access=False,target_evaluated=False)
        cfg=dict(method='csil',source_counts=dict(L_s=6300,U_s=56700,V=27000))
        p.provenance('csil',c,c,init,done,cfg)
        for where,key,value in [(1,'role_ids',{}),(1,'classes',list(reversed(p.CLASSES))),
                (2,'ancestors',['unknown']),(2,'selection','best_target'),(2,'scratch_only',False),
                (2,'target_contact',True),(3,'target_evaluated',True),(3,'epoch',199),(4,'method','mopc_hr')]:
            args=[deepcopy(x) for x in ['csil',c,c,init,done,cfg]]
            args[where+1][key]=value
            with self.assertRaises(ValueError):p.provenance(*args)

if __name__=='__main__':unittest.main()
