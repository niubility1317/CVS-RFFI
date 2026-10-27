import unittest
import numpy as np
from experiments.pamf_channel_pl import rx_consensus as m
from experiments.pamf_channel_pl import rx_consensus_source as runner
from tests.test_pamf_receiver_balance_source import context,runtime


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.rx=[1,3,4,6,8];self.x=np.tile(np.eye(6,dtype=np.float32),(5,1));self.y=np.tile(np.arange(6),5)
        self.r=np.repeat(self.rx,6);self.p,self.counts=m.bank(self.x,self.y,self.r,self.rx)

    def test_L_only_bank_and_all_cells(self):
        np.testing.assert_array_equal(self.counts,np.ones((5,6)))
        for role in ('U_s','V','target'):
            with self.assertRaises(ValueError):m.bank(self.x,self.y,self.r,self.rx,role)
        with self.assertRaises(ValueError):m.bank(self.x[:-1],self.y[:-1],self.r[:-1],self.rx)

    def test_nonfinite_zero_unknown_receiver_or_hidden_label_rejected(self):
        for x in (np.zeros_like(self.x),self.x*np.nan):
            with self.assertRaises(ValueError):m.bank(x,self.y,self.r,self.rx)
        for y,r in ((self.y-1,self.r),(self.y,self.r+100)):
            with self.assertRaises(ValueError):m.bank(self.x,y,r,self.rx)

    def decide(self,q,p=None,x=None):
        return m.decisions(np.array([[1,0,0,0,0,0]],dtype=np.float32) if x is None else x,
            np.array([q],dtype=np.float32),np.array([1]),self.p if p is None else p,self.rx)

    def test_own_receiver_never_votes_or_controls_margin(self):
        p=self.p.copy();p[0]=np.roll(p[0],1,axis=0)
        d=self.decide([2.8,0,-5,-5,-5,-5],p)
        self.assertTrue(d['A_same_top1'][0]);self.assertEqual(d['consensus_label'][0],0)
        p[1]=np.roll(p[1],1,axis=0)
        self.assertFalse(self.decide([2.8,0,-5,-5,-5,-5],p)['consensus'][0])

    def test_base95_preserved_and_B_only_relabels_top2(self):
        d=self.decide([10,0,-5,-5,-5,-5]);self.assertTrue(d['base_accepted'][0]);self.assertFalse(d['B_within_top2'][0])
        d=self.decide([0,2.8,-5,-5,-5,-5]);self.assertFalse(d['base_accepted'][0]);self.assertFalse(d['A_same_top1'][0]);self.assertTrue(d['B_within_top2'][0])
        d=self.decide([-5,2.8,0,-5,-5,-5]);self.assertFalse(d['B_within_top2'][0])

    def test_mass_margin_and_A_subset(self):
        d=self.decide([0,0,0,0,0,0]);self.assertFalse(d['B_within_top2'][0])
        x=np.array([[1,1,0,0,0,0]],dtype=np.float32)
        d=self.decide([2.8,0,-5,-5,-5,-5],x=x);self.assertFalse(d['consensus'][0])
        d=self.decide([2.8,0,-5,-5,-5,-5]);self.assertTrue(d['A_same_top1'][0]);self.assertTrue(d['B_within_top2'][0])

    def test_frozen_exact_source_gate_and_no_auto_target_permission(self):
        y=np.zeros(27000,dtype=np.int64);base=np.zeros(27000,dtype=bool);extra=base.copy();extra[:135]=True
        pred=dict(base_accepted=base,base_label=y.copy(),consensus_label=y.copy(),A_same_top1=extra,B_within_top2=extra.copy())
        out=m.score(pred,y);self.assertEqual(out['selected_for_later_source_pilot'],m.RULES[0]);self.assertEqual(out['target_reads'],0)
        pred['consensus_label'][:2]=1;self.assertIsNone(m.score(pred,y)['selected_for_later_source_pilot'])
        pred['consensus_label'][:]=0;pred['A_same_top1'][134]=False;pred['B_within_top2'][134]=False
        self.assertIsNone(m.score(pred,y)['selected_for_later_source_pilot'])
        with self.assertRaises(ValueError):m.score(pred,y[:100])

    def test_context_runtime_reject_G_target_and_foreign_roots(self):
        c=context();runner.validate_context(c)
        for key in ('G','target_inputs','truth'):
            with self.assertRaises(ValueError):runner.validate_context(dict(c,**{key:'forbidden'}))
        rt=runtime();rt.update(run_id=runner.RUN,launch_owner=runner.OWNER,run_root='/project/runs/'+runner.RUN,
            release_root='/project/releases/'+runner.RELEASE,context='/project/releases/'+runner.RELEASE+'/context.json',
            runtime_file='/project/releases/'+runner.RELEASE+'/runtime.json')
        runner.validate_runtime(rt)
        for change in (dict(target_inputs={}),dict(allowed_gpus=[0]),dict(run_root='/foreign')):
            with self.assertRaises(ValueError):runner.validate_runtime(dict(rt,**change))

    def test_role_loader_never_accepts_U_or_target(self):
        for role in ('U_s','target','query'):
            with self.assertRaises(ValueError):runner.role_input({}, {},role)

    def test_separate_scorer_rejects_F_G_or_target_handles_before_any_io(self):
        for key in ('F','G','target_truth'):
            with self.assertRaises(ValueError):runner.score_source(dict(run_root='/runs/'+runner.RUN,source_cache='unused',source_contract='unused',**{key:'forbidden'}))


if __name__=='__main__':unittest.main()
