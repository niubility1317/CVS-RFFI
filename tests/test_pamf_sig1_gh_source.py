import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from pathlib import Path
import torch
from experiments.pamf_channel_pl import sig1_gh_source as run
from experiments.pamf_channel_pl import sig1_repl_offset_source as old
from experiments.pamf_channel_pl.sig1_gh_objective import GHObjective,rejected_peer_loss,PAIR_WEIGHT
from experiments.pamf_channel_pl.sig1_role_objective import RestoredRoles,mask
from tests.test_pamf_sig1_role_training import config as role_config


def config(method):
    base=role_config('REPL_G_U')
    with patch.object(old,'config',return_value=base):return run.config(dict(project='/project'),'commit',method)


class GHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_detached_peer_exact_gradients_and_full_half_normalization(self):
        a=torch.randn(8,6,dtype=torch.double,requires_grad=True);b=torch.randn(8,6,dtype=torch.double,requires_grad=True)
        rejected=torch.tensor([1,0,1,0,0,0,0,0],dtype=torch.bool)
        ga,gb=torch.autograd.grad(rejected_peer_loss(a,b,rejected),(a,b));expected=.5*(a.softmax(1)-b.softmax(1))*rejected[:,None]/8
        self.assertTrue(torch.allclose(ga,expected,atol=1e-14));self.assertTrue(torch.allclose(gb,-expected,atol=1e-14))
        self.assertTrue(bool((ga[~rejected]==0).all()));self.assertAlmostEqual(PAIR_WEIGHT,1.2)

    def test_shared_wrong_prediction_has_no_consistency_gradient(self):
        a=torch.tensor([[0.,9.,0,0,0,0]],requires_grad=True);b=a.detach().clone().requires_grad_()
        ga,gb=torch.autograd.grad(rejected_peer_loss(a,b,torch.ones(1,dtype=torch.bool)),(a,b))
        self.assertLess(float(ga.abs().max()),1e-7);self.assertLess(float(gb.abs().max()),1e-7)

    def test_no_rejected_packets_gives_zero_finite_gradients(self):
        a=torch.randn(4,6,requires_grad=True);b=torch.randn(4,6,requires_grad=True)
        loss=rejected_peer_loss(a,b,torch.zeros(4,dtype=torch.bool));loss.backward()
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(float(a.grad.abs().sum()+b.grad.abs().sum()),0.)

    def test_pair_rejects_shape_and_nonfinite(self):
        a=torch.randn(4,6);gate=torch.ones(4,dtype=torch.bool)
        for first,second,m in ((a,a[:2],gate),(a,a,gate.float()),(a*float('nan'),a,gate)):
            with self.assertRaises(ValueError):rejected_peer_loss(first,second,m)

    def test_existing_mixed_H_half_is_generated_once_and_preserved(self):
        obj=GHObjective.__new__(GHObjective);obj.c=dict(hard_pl_no_uas=False,candidate_smoke=True);obj.H_calls=obj.H_packets=0
        x=torch.randn(4,2,256);ordinary=x+1;physical=mask(4,4441,'cpu');meta=dict(rx_i=torch.arange(4));calls=[]
        def H(x,meta,epoch,step):calls.append(len(x));return x+3
        value=obj.mixed_view(SimpleNamespace(_cd_practical=H),x,ordinary,physical,meta,21,4441,None,None)
        self.assertEqual(calls,[2]);self.assertTrue(torch.equal(value[physical],x[physical]+3));self.assertTrue(torch.equal(value[~physical],ordinary[~physical]))
        self.assertEqual((obj.H_calls,obj.H_packets),(1,2));self.assertGreater(obj.smoke_H_relative_difference,0)

    def test_G_pair_captures_only_two_existing_mixed_logits(self):
        obj=GHObjective.__new__(GHObjective);obj._joint_forward_index=0;obj.c=dict(step_offset=4440);obj.pair_captures=[]
        x=torch.randn(4,2,256);logits=torch.randn(4,6);accepted=torch.tensor([True,False,False,True]);student=SimpleNamespace(_feature_l_steps=1)
        def parent(self,*args):self._joint_forward_index+=1;return logits
        with patch.object(RestoredRoles,'unlabeled_student_forward',parent):
            for _ in range(6):obj.unlabeled_student_forward(student,x,None,None,accepted,{})
        self.assertEqual(len(obj.pair_captures),2);physical=mask(4,4441,'cpu')
        self.assertTrue(torch.equal(obj.pair_captures[0][0],logits[~physical]));self.assertTrue(torch.equal(obj.pair_captures[0][1],~accepted[~physical]))

    def test_two_configs_require_H_and_exact_rejected_pair_recipe(self):
        for method in run.METHODS:
            c=config(method);run.validate_config(c);self.assertFalse(c['hard_pl_no_uas']);self.assertFalse(c['restored_roles']['enable_L']);self.assertTrue(c['restored_roles']['enable_U'])
            self.assertEqual(c['GH_pair']['enabled'],method.endswith('_Pair'))
            for change in (dict(hard_pl_no_uas=True),dict(tests={}),dict(sig_time_views=3),dict(own_EMA=True),dict(posthoc_head_fit=True)):
                with self.assertRaises(ValueError):run.validate_config(dict(c,**change))
            bad=deepcopy(c);bad['GH_pair']['raw_rejected_only']=False
            with self.assertRaises(ValueError):run.validate_config(bad)

    def test_exact_source_gate_prefers_no_pair_at_tie(self):
        refs={m:dict(raw_correct=27000,H3_correct=81000,G3_correct=75000) for m in run.REFERENCES}
        scores=dict(refs,**{m:dict(raw_correct=26973,H3_correct=80919,G3_correct=75243) for m in run.METHODS})
        self.assertEqual(run.choose_source(scores)[1],'REPL_GH_U')
        for key in ('raw_correct','H3_correct','G3_correct'):
            bad=deepcopy(scores)
            for method in run.METHODS:bad[method][key]-=1
            self.assertIsNone(run.choose_source(bad)[1])

    def test_prepared_smoke_and_students_execute_GH_module(self):
        ctx=dict(cases=[dict(model_seed=run.SEED,expected_source_contract='roles')])
        source=dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,U_truth_read=False,head_fit_updates=0,config=dict(source_contract='roles'))
        def reader(path):
            value=str(path).replace('\\','/')
            if value=='context':return ctx
            if value=='roles':return dict(ids='existing')
            if value=='source/complete.json':return source
            if value.endswith('complete.json'):return dict(source_output='source')
            raise AssertionError(value)
        with tempfile.TemporaryDirectory() as folder:
            rt=dict(context='context',run_root=str(Path(folder)/'run'),project='/project',code_commit='c',release_root='/release',python='python')
            with patch.object(run,'read',reader),patch.object(run,'validate_context'),patch.object(run,'source_score',return_value=dict(raw_correct=1,H3_correct=2,G3_correct=3)),patch.object(run,'config',side_effect=lambda ctx,commit,m:dict(paper_method=m)):
                jobs=run.prepare(rt)
            self.assertEqual({j['entrypoint'] for j in jobs},{'experiments.pamf_channel_pl.sig1_gh_source'})
            self.assertEqual({j['id'] for j in jobs},{'smoke',*run.METHODS})


if __name__=='__main__':unittest.main()
