from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_channel_pl.residual_mix_training import config,validate_config,Objective,VARIANTS,ROUTES,BASE_WEIGHT,prepare
from experiments.pamf_channel_pl.residual_mix_calibration import decide


def cfg(method):
    original=dict(run_id='original',row_id='F0-ordinary_PL-s2026092701',model_seed=2026092701,num_classes=6,
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='source',source_contract='roles',ablation='F21')
    case=dict(protocol='F0',model_seed=2026092701,row_id=original['row_id'],parent_source='parent/segment20',expected_source_contract='roles',source_config=original)
    with patch('experiments.pamf_channel_pl.residual_mix_training.read',return_value=dict(config=original)):
        c=config(case,'project','commit',method)
    c['source_G']=dict(kind='original_F0_LU',roles='LU')
    return c


class ResidualMixTrainingTests(unittest.TestCase):
    def test_fixed_source_protocol_budget_and_mixture_recipe(self):
        for method in VARIANTS:
            c=validate_config(cfg(method));self.assertEqual(c['residual_mix']['route'],ROUTES[method])
            self.assertTrue(c['residual_mix']['shield_classifier_weight']);self.assertTrue(c['default_head_joint'])
            self.assertEqual(c['residual_mix']['weight_rule'],'base' if method.endswith('Base') else 'source_route_10pct')
            for key,value in [('tests',{}),('target_inputs','query'),('model_seed',2026092702),('segment_epochs',21),
                              ('default_head_joint',False),('parent_source','other/segment30'),('own_EMA',True)]:
                bad=deepcopy(c);bad[key]=value
                with self.assertRaises(ValueError):validate_config(bad)
            for key,value in [('teacher_updates',1),('new_head',True),('held_extra_rx',3),('alpha','other'),
                              ('mix_seed',0),('base_weight',.2),('private_random_draws_per_U',1)]:
                bad=deepcopy(c);bad['residual_mix'][key]=value
                with self.assertRaises(ValueError):validate_config(bad)

    def test_calibration_and_real_smoke_are_job_dependencies(self):
        with TemporaryDirectory() as root:
            context=dict(cases=[dict(model_seed=2026092701)],G=dict(kind='original_F0_LU',roles='LU'))
            rt=dict(context='context',run_root=root,project='project',code_commit='commit',frozen_sources={},release_root='release',python='python')
            with patch('experiments.pamf_channel_pl.residual_mix_training.read',return_value=context), \
                    patch('experiments.pamf_channel_pl.residual_mix_training.validate_context'), \
                    patch('experiments.pamf_channel_pl.residual_mix_training.config',side_effect=lambda case,project,commit,method:cfg(method)):
                jobs=prepare(rt)
            self.assertEqual(len(jobs),6);by={j['id']:j for j in jobs}
            self.assertEqual(by['smoke']['dependencies'],[by['source_calibration']['completion']])
            self.assertTrue(all(by[v]['dependencies']==[by['smoke']['completion']] for v in VARIANTS))
            self.assertEqual(read(read(Path(root)/'configs/smoke.json')['train_config'])['paper_method'],'PairDose')
            self.assertTrue(all(j['entrypoint']=='experiments.pamf_channel_pl.residual_mix_training' for j in jobs))

    def test_hidden_U_views_exposure_private_rng_and_feature_gradient(self):
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.head=torch.nn.Linear(160,6,bias=False)
            def classify_features(self,z):return self.head(z)*.01
        class Student(torch.nn.Module):
            def __init__(self,batch):
                super().__init__();self.feature=torch.nn.Linear(512,160);self.id_backbone=torch.nn.Module()
                self.id_backbone.encoder=Encoder();self.inputs=[]
                self._cd_U_loader=[batch];self._cd_U_iter=iter([batch]);self._cd_U_seen=self._cd_U_accepted=0
            def forward_identity_only(self,x,domain_labels=None):
                self.inputs.append(x.detach().clone());z=self.feature(x.flatten(1))
                return dict(z_id=z,tx_logits=self.id_backbone.encoder.classify_features(z))
            def forward(self,x):return self.forward_identity_only(x)['tx_logits']
        class Cache:
            def __init__(self):self.calls=[]
            def take(self,metadata,k,device,with_IQ):
                self.calls.append((k,with_IQ));q=torch.nn.functional.one_hot(torch.tensor([0,1,2,3]),6).float()
                return q,torch.ones(4,dtype=torch.bool),batch[0]+1.5
        meta=dict(rx_i=torch.tensor([1,1,3,8]),tx_label_visible=torch.zeros(4,dtype=torch.bool))
        batch=(torch.randn(4,2,256),torch.full((4,),-1),torch.zeros(4,dtype=torch.long),meta)
        inputs={}
        for method in VARIANTS:
            torch.manual_seed(42)
            o=Objective.__new__(Objective);o.c=cfg(method);o.cache=Cache();o.weight=BASE_WEIGHT if method.endswith('Base') else .2;o.metrics={};o.generator=None
            o.calls=o.seen=o.raw_accepted=o.extra_accepted=o.held_seen=o.mix_forwards=o.G_forwards=o.alpha_count=0
            o.alpha_sum=o.alpha_square_sum=0.;o.alpha_min=1.;o.alpha_max=0.;o.alpha_bins=[0,0,0,0];o.expected_extra_per_epoch=2
            student=Student(batch);rng=torch.random.get_rng_state().clone()
            value=o.loss(student,'cpu');value.backward()
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(o.extra_accepted,2);self.assertEqual(o.held_seen,2);self.assertEqual(o.cache.calls,[(0,True)])
            self.assertEqual(o.raw_accepted,0);self.assertEqual(float(student.id_backbone.encoder.head.weight.grad.norm()),0.)
            self.assertGreater(float(student.feature.weight.grad.norm()),0.)
            mixtures=2 if ROUTES[method]=='MixMix' else 1;state=o.state_dict()
            self.assertEqual(state['mix_forwards'],4*mixtures);self.assertEqual(state['G_endpoint_forwards'],4*(2-mixtures))
            self.assertEqual(state['alpha_count'],4*mixtures);self.assertEqual(sum(state['alpha_bins']),4*mixtures)
            self.assertEqual(state['private_random_draws'],8);self.assertTrue(0<=state['alpha_min']<=state['alpha_max']<1)
            if mixtures==1:self.assertTrue(torch.equal(student.inputs[-1],batch[0]+1.5))
            inputs[method]=student.inputs[-2:]
            meta['tx_label_visible'][0]=True;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            meta['tx_label_visible'][0]=False;batch[1][0]=0;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            batch[1][0]=-1
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(inputs['MixBase'],inputs['MixDose'])))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(inputs['PairBase'],inputs['PairDose'])))
        self.assertTrue(torch.equal(inputs['MixBase'][0],inputs['PairBase'][0]))

    def test_source_dose_rule_has_route_specific_cap_and_no_target(self):
        row=dict(primary_norm=1.,MixMix_norm=.2,MixG_norm=.8,GEndpoint_norm=1.6,
                 MixMix_cosine=.1,MixG_cosine=-.2,GEndpoint_cosine=-.3)
        rows=[dict(row) for _ in range(12)];d=decide(rows)
        self.assertEqual(d['weights'],dict(MixMix=.25,MixG=.125))
        self.assertEqual(d['dose_upper_cap_active'],dict(MixMix=True,MixG=False))
        self.assertLessEqual(max(d['weighted_median_norm_ratio'].values()),.1+1e-12)
        with self.assertRaises(ValueError):decide(rows[:11])
        rows[0]['MixMix_norm']=float('nan')
        with self.assertRaises(ValueError):decide(rows)


if __name__=='__main__':unittest.main()
