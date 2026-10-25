from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_channel_pl.feature_task_training import config,validate_config,Objective,VARIANTS,ROUTES,prepare
from experiments.pamf_channel_pl.feature_task_calibration import decide


def cfg(method):
    original=dict(run_id='original',row_id='F0-ordinary_PL-s2026092701',model_seed=2026092701,num_classes=6,
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='source',source_contract='roles',ablation='F21')
    case=dict(protocol='F0',model_seed=2026092701,row_id=original['row_id'],parent_source='parent/segment20',expected_source_contract='roles',source_config=original)
    with patch('experiments.pamf_channel_pl.feature_task_training.read',return_value=dict(config=original)):
        c=config(case,'project','commit',method)
    c['source_G']=dict(kind='original_F0_LU',roles='LU')
    return c


class FeatureTaskTrainingTests(unittest.TestCase):
    def test_source_protocol_and_feature_routing_cannot_be_changed(self):
        for method in VARIANTS:
            c=validate_config(cfg(method))
            self.assertEqual(c['feature_task']['views'],list(ROUTES[method]))
            self.assertEqual(c['feature_task']['shield_classifier_weight'],method!='GJointHard')
            self.assertTrue(c['default_head_joint'])
            for key,value in [('tests',{}),('target_inputs','query'),('model_seed',2026092702),('segment_epochs',19),
                              ('default_head_joint',False),('parent_source','other/segment30'),('own_EMA',True)]:
                bad=deepcopy(c);bad[key]=value
                with self.assertRaises(ValueError):validate_config(bad)
            for key,value in [('teacher_updates',1),('new_head',True),('held_extra_rx',3),('ramp_steps',0),('views',['G','H','H'])]:
                bad=deepcopy(c);bad['feature_task'][key]=value
                with self.assertRaises(ValueError):validate_config(bad)

    def test_job_graph_requires_real_calibration_and_integrated_smoke(self):
        with TemporaryDirectory() as root:
            context=dict(cases=[dict(model_seed=2026092701)],G=dict(kind='original_F0_LU',roles='LU'))
            rt=dict(context='context',run_root=root,project='project',code_commit='commit',frozen_sources={},release_root='release',python='python')
            with patch('experiments.pamf_channel_pl.feature_task_training.read',return_value=context), \
                    patch('experiments.pamf_channel_pl.feature_task_training.validate_context'), \
                    patch('experiments.pamf_channel_pl.feature_task_training.config',side_effect=lambda case,project,commit,method:cfg(method)):
                jobs=prepare(rt)
            self.assertEqual(len(jobs),6)
            by={j['id']:j for j in jobs}
            self.assertEqual(by['smoke']['dependencies'],[by['source_calibration']['completion']])
            self.assertTrue(all(by[v]['dependencies']==[by['smoke']['completion']] for v in VARIANTS))
            self.assertEqual(read(read(Path(root)/'configs/smoke.json')['train_config'])['paper_method'],'HGFeatureHard')
            self.assertTrue(all(j['entrypoint']=='experiments.pamf_channel_pl.feature_task_training' for j in jobs))

    def test_actual_hidden_U_loss_has_expected_views_holdout_and_weight_shield(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.head=torch.nn.Linear(160,6,bias=False)
            def classify_features(self,z):return self.head(z)*.01
        class Student(torch.nn.Module):
            def __init__(self,batch):
                super().__init__();self.feature=torch.nn.Linear(160,160);self.gain=torch.nn.Parameter(torch.ones(160))
                self.id_backbone=torch.nn.Module();self.id_backbone.encoder=Encoder();self.inputs=[]
                self._cd_U_loader=[batch];self._cd_U_iter=iter([batch]);self._cd_U_seen=self._cd_U_accepted=0
            def forward_identity_only(self,x,domain_labels=None):
                self.inputs.append(x.detach().clone());z=self.feature(x)*self.gain
                return dict(z_id=z,tx_logits=self.id_backbone.encoder.classify_features(z))
            def forward(self,x):return self.forward_identity_only(x)['tx_logits']
        class Cache:
            def __init__(self):self.calls=[]
            def take(self,metadata,k,device):
                self.calls.append(k);q=torch.nn.functional.one_hot(torch.tensor([0,1,2,3]),6).float()
                return q,torch.ones(4,dtype=torch.bool),torch.ones(4,160),torch.ones(4,160)*2
        meta=dict(rx_i=torch.tensor([1,1,3,8]),tx_label_visible=torch.zeros(4,dtype=torch.bool))
        batch=(torch.randn(4,160),torch.full((4,),-1),torch.zeros(4,dtype=torch.long),meta)
        for method in VARIANTS:
            torch.manual_seed(42)
            o=Objective.__new__(Objective);o.c=cfg(method);o.cache=Cache();o.weights=dict(G=.04,H=.02);o.metrics={}
            o.calls=o.seen=o.raw_accepted=o.extra_accepted=o.held_seen=o.H_forwards=o.G_forwards=0;o.expected_extra_per_epoch=2
            student=Student(batch);value=o.loss(student,'cpu');value.backward()
            self.assertEqual(o.extra_accepted,2);self.assertEqual(o.held_seen,2);self.assertEqual(o.seen,4);self.assertEqual(o.cache.calls,[0])
            for actual,view in zip(student.inputs[-2:],ROUTES[method]):
                self.assertTrue(torch.equal(actual,torch.ones(4,160)*(1 if view=='H' else 2)))
            self.assertEqual(o.raw_accepted,0)
            head_gradient=student.id_backbone.encoder.head.weight.grad
            if method=='GJointHard':self.assertGreater(float(head_gradient.norm()),0.)
            else:self.assertEqual(float(head_gradient.norm()),0.)
            self.assertGreater(float(student.feature.weight.grad.norm()),0.)
            self.assertGreater(float(student.gain.grad.norm()),0.)
            state=o.state_dict();self.assertEqual(state['extra_student_forwards'],8)
            self.assertEqual(state['H_forwards'],4*ROUTES[method].count('H'));self.assertFalse(state['new_head'])
            meta['tx_label_visible'][0]=True;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            meta['tx_label_visible'][0]=False;batch[1][0]=0;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            batch[1][0]=-1

    def test_calibration_shares_G_coefficient_and_bounds_source_proxy(self):
        row=dict(primary_norm=2.,GJointHard_norm=4.,GFeatureHard_norm=3.,HFeatureHard_norm=8.,
                 GJointHard_cosine=-.1,GFeatureHard_cosine=.2,HFeatureHard_cosine=.1)
        rows=[dict(row) for _ in range(12)];d=decide(rows)
        self.assertEqual(d['weights'],dict(G=.05,H=.025))
        self.assertLessEqual(max(d['weighted_median_norm_ratio'].values()),.1+1e-12)
        with self.assertRaises(ValueError):decide(rows[:11])
        rows[0]['GJointHard_norm']=float('nan')
        with self.assertRaises(ValueError):decide(rows)


if __name__=='__main__':unittest.main()
