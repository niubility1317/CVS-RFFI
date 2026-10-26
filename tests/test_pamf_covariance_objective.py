from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.pamf_u_history.covariance_objective import Objective,combine,schedule,RECIPE
from experiments.pamf_u_history.covariance_source import validate_config,RUN
from experiments.pamf_u_history.source import prefix


class TinyStudent(nn.Module):
    def __init__(self,batch,data):
        super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module();self.id_backbone.encoder.core=nn.Module()
        self.id_backbone.encoder.core.id_backbone=nn.Module();b=self.id_backbone.encoder.core.id_backbone
        b.time_down=nn.Conv1d(2,4,1);b.t2=nn.Module();b.t2.pool=nn.AvgPool1d(2);b.head=nn.Linear(4,6);self.drop=nn.Dropout(.2)
        self._cd_U_loader=SimpleNamespace(dataset=data);self._cd_U_iter=iter([batch]);self._cd_U_seen=self._cd_U_accepted=0;self._feature_l_steps=4441
    def forward_identity_only(self,x,domain_labels=None):
        b=self.id_backbone.encoder.core.id_backbone;z=b.t2.pool(b.time_down(x)).mean(-1)
        return dict(tx_logits=b.head(self.drop(z)),z_id=z)
    def forward(self,x):return self.forward_identity_only(x)['tx_logits']


class ObjectiveTest(unittest.TestCase):
    def test_schedule_complete_replica_budget(self):
        count=[0,0,0]
        for call in range(1,6661):
            r,k=schedule(call);count[r-1]+=1;self.assertIn(k,(0,1,2))
        self.assertEqual(count,[2220,2220,2220])

    def test_native_L_and_PL_reuse_allU_scope_and_rng(self):
        torch.manual_seed(9);n=56700;rx=torch.tensor([1,3,4,6,8])[torch.arange(n)%5]
        meta=dict(eq_i=torch.ones(n,dtype=torch.long),rx_i=rx,day_i=torch.ones(n,dtype=torch.long),sig_i=torch.arange(n),base_index=torch.arange(n))
        data=SimpleNamespace(role='U_s',metadata=meta)
        small={k:v[:256] for k,v in meta.items()};small['tx_label_visible']=torch.zeros(256,dtype=torch.bool)
        batch=(torch.randn(256,2,8),torch.full((256,),-1,dtype=torch.long),torch.zeros(256,dtype=torch.long),small)
        m=TinyStudent(batch,data);m.train();x=torch.randn(128,2,8)
        masks={level:{'selective':torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1),'random':torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1),'full':torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1)} for level in ('time_down','t2.pool')}
        c=dict(covariance_calibration=True,covariance_coefficient=.2,paper_method='CovSel-U',source_rxs=[1,3,4,6,8])
        class Cache:
            def __init__(self,*a):pass
            def take(self,*a):return batch[0]+.4*batch[0].sin().square()
        with patch('experiments.pamf_u_history.covariance_objective.load_masks',return_value=masks),patch('experiments.pamf_u_history.covariance_objective.SourceViews',Cache):
            obj=Objective(c)
            with obj.observe(m):
                for value in (x,x+.3*x.sin().square(),x,x,x):m.forward_identity_only(value)
                main=obj.plain(m,'cpu');rng=torch.random.get_rng_state().clone()
                extra=obj.extra(m,dict(labels=torch.zeros(128,dtype=torch.long),extra=(None,dict(rx_i=rx[:128]))))
                self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(obj.L_forwards,5);self.assertEqual(obj.history.state()['covariance_U_seen'],256)
            self.assertEqual(m._cd_U_seen,256);self.assertEqual(obj.history.accepted,0)
            self.assertTrue(torch.equal(obj.candidates['CovSel-U'],obj.candidates['CovSel-LU']))
            nodes=[obj.maps['L_raw']['time_down'],obj.maps['U_raw']['time_down']]
            gu=torch.autograd.grad(obj.candidates['CovSel-U'],nodes,allow_unused=True,retain_graph=True)
            gl=torch.autograd.grad(obj.candidates['CovSel-LU'],nodes,allow_unused=True,retain_graph=True)
            self.assertIsNone(gu[0]);self.assertGreater(float(gu[1].norm()),0);self.assertGreater(float(gl[0].norm()),0)
            (main+extra).backward();self.assertTrue(torch.isfinite(m.id_backbone.encoder.core.id_backbone.time_down.weight.grad).all())
            self.assertTrue(all(len(layer._forward_hooks)==0 for layer in m.modules()))


class SourceConfigTest(unittest.TestCase):
    def config(self):
        method='CovSel-U';rid='F0-'+method+'-s2026092701'
        return dict(run_id=RUN,row_id=rid,paper_method=method,model_seed=2026092701,project='/p',
            algorithm_id='pamf_sig1_selective_covariance_v1',method_id='pamf_t1_e50',covariance_recipe=RECIPE,
            covariance_masks='/p/runs/'+RUN+'/mask_preparation/masks.pt',covariance_coefficient=.2,covariance_calibration=False,
            num_classes=6,source_rxs=[1,3,4,6,8],source_days=[1,2,3],sig_time_views=1,epochs=50,optimizer_updates=11100,trajectory=[20,10,20],
            shared_SIG1_E20=prefix('/p'),regime='matched_sat_plain_pl',ablation='F21',U_use=True,U_start_epoch=14,
            plain_PL_threshold=.95,plain_PL_weight=1.,labeled_batch=128,unlabeled_batch=256,default_head_joint=True,
            repl_row=dict(structure='SIG_TIME',L_A=True,L_B=True,U=True,blocks={}),output_root='/p/runs/'+RUN+'/F/'+rid+'/source')
    def test_original_PL_and_source_permissions_cannot_change(self):
        c=self.config();validate_config(c);c['plain_PL_weight']=2
        with self.assertRaises(ValueError):validate_config(c)
        c=self.config();c['truth']='foreign'
        with self.assertRaises(ValueError):validate_config(c)
    def test_no_shortened_budget_or_calibration_training_coefficient(self):
        c=self.config();c.update(segment_epochs=1,epoch_offset=20,step_offset=4440,parent_source=c['shared_SIG1_E20'])
        with self.assertRaises(ValueError):validate_config(c)
        c=self.config();c['covariance_calibration']=True
        with self.assertRaises(ValueError):validate_config(c)


if __name__=='__main__':unittest.main()
