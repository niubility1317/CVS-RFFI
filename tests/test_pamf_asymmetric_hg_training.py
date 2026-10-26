from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_channel_pl.asymmetric_hg_training import config,validate_config,Objective,VARIANTS,ROUTES,prepare
from experiments.pamf_channel_pl.asymmetric_hg_calibration import decide


def cfg(method):
    original=dict(run_id='original',row_id='F0-ordinary_PL-s2026092701',model_seed=2026092701,num_classes=6,
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='source',source_contract='roles',ablation='F21')
    case=dict(protocol='F0',model_seed=2026092701,row_id=original['row_id'],parent_source='parent/segment20',expected_source_contract='roles',source_config=original)
    with patch('experiments.pamf_channel_pl.asymmetric_hg_training.read',return_value=dict(config=original)):
        c=config(case,'project','commit',method)
    c['source_G']=dict(kind='original_F0_LU',roles='LU');return c


class AsymmetricHGTrainingTests(unittest.TestCase):
    def test_protocol_recipe_and_real_smoke_dependency(self):
        for method in VARIANTS:
            c=validate_config(cfg(method));self.assertEqual(c['asymmetric_hg']['views'],list(ROUTES[method]))
            for key,value in [('tests',{}),('target_inputs','query'),('model_seed',2026092702),('segment_epochs',21),
                              ('default_head_joint',False),('parent_source','other/segment30'),('own_EMA',True)]:
                bad=deepcopy(c);bad[key]=value
                with self.assertRaises(ValueError):validate_config(bad)
            bad=deepcopy(c);bad['asymmetric_hg']['temperature']=.1
            with self.assertRaises(ValueError):validate_config(bad)
        with TemporaryDirectory() as folder:
            context=dict(cases=[dict(model_seed=2026092701)],G=dict(kind='original_F0_LU',roles='LU'))
            rt=dict(context='context',run_root=folder,project='project',code_commit='commit',frozen_sources={},release_root='release',python='python')
            with patch('experiments.pamf_channel_pl.asymmetric_hg_training.read',return_value=context), \
                    patch('experiments.pamf_channel_pl.asymmetric_hg_training.validate_context'), \
                    patch('experiments.pamf_channel_pl.asymmetric_hg_training.config',side_effect=lambda case,project,commit,method:cfg(method)):
                jobs=prepare(rt)
            by={j['id']:j for j in jobs};self.assertEqual(len(jobs),6)
            self.assertEqual(by['smoke']['dependencies'],[by['source_calibration']['completion']])
            self.assertTrue(all(by[v]['dependencies']==[by['smoke']['completion']] for v in VARIANTS))
            self.assertEqual(read(read(Path(folder)/'configs/smoke.json')['train_config'])['paper_method'],'BothJoint')

    def test_all_hidden_U_routes_have_two_feature_passes_and_no_extra_head_gradient(self):
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.head=torch.nn.Linear(160,6,bias=False)
            def classify_features(self,z):return self.head(z)*.001
        class Student(torch.nn.Module):
            def __init__(self,batch):
                super().__init__();self.feature=torch.nn.Linear(512,160);self.id_backbone=torch.nn.Module();self.id_backbone.encoder=Encoder();self.inputs=[]
                self._cd_U_loader=[batch];self._cd_U_iter=iter([batch]);self._cd_U_seen=self._cd_U_accepted=0
            def forward_identity_only(self,x,domain_labels=None):
                self.inputs.append(x.detach().clone());z=self.feature(x.flatten(1))
                return dict(z_id=z,tx_logits=self.id_backbone.encoder.classify_features(z))
            def forward(self,x):return self.forward_identity_only(x)['tx_logits']
        class Cache:
            def __init__(self):self.base=self;self.calls=[]
            def take(self,metadata,k,device,*args):
                self.calls.append(k);q=torch.nn.functional.one_hot(torch.tensor([0,0,1,1,2,2,0,1]),6).float();mask=torch.ones(8,dtype=torch.bool)
                g=batch[0]*(1+.1*(k+1))+.05
                return (q,mask,g) if args else (q,mask,batch[0]+.3,g)
        meta=dict(rx_i=torch.tensor([3,4,3,4,3,4,1,1]),tx_label_visible=torch.zeros(8,dtype=torch.bool))
        batch=(torch.randn(8,2,256),torch.full((8,),-1),torch.zeros(8,dtype=torch.long),meta)
        pair_counts=[];route_inputs={}
        for method in VARIANTS:
            torch.manual_seed(42);o=Objective.__new__(Objective);o.c=cfg(method);o.cache=Cache();o.weight=.12;o.metrics={}
            for key in ('calls','seen','raw_accepted','extra_accepted','held_seen','raw_forwards','H_forwards','G_forwards','valid_anchors','positive_pairs','negative_pairs','CE_H_accepted','CE_G_accepted'):setattr(o,key,0)
            o.expected_extra_per_epoch=6;student=Student(batch);value=o.loss(student,'cpu');value.backward();state=o.state_dict()
            self.assertEqual(o.extra_accepted,6);self.assertEqual(o.held_seen,2);self.assertEqual(len(student.inputs),4)
            self.assertEqual(o.raw_accepted,0);self.assertEqual(float(student.id_backbone.encoder.head.weight.grad.norm()),0)
            self.assertGreater(float(student.feature.weight.grad.norm()),0)
            self.assertEqual(state['extra_student_forwards'],16);self.assertEqual(state['valid_anchors'],12)
            self.assertEqual(o.cache.calls,[0]);pair_counts.append((o.positive_pairs,o.negative_pairs))
            self.assertEqual(state['CE_H_accepted'],6 if method in ('HJoint','HAnchor','BothJoint') else 0)
            self.assertEqual(state['CE_G_accepted'],6 if method in ('GJoint','BothJoint') else 0)
            self.assertEqual(state['restored_feature_detached_in_CL'],method=='HAnchor')
            route_inputs[method]=student.inputs[-2:]
            meta['tx_label_visible'][0]=True;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            meta['tx_label_visible'][0]=False;batch[1][0]=0;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            batch[1][0]=-1
        self.assertTrue(all(c==pair_counts[0] for c in pair_counts))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(route_inputs['HJoint'],route_inputs['HAnchor'])))

    def test_common_source_dose_uses_largest_method_median(self):
        row=dict(primary_norm=1.)
        for i,m in enumerate(VARIANTS):row.update({m+'_norm':float(i+1),m+'_cosine':-.1})
        rows=[dict(row) for _ in range(12)];d=decide(rows)
        self.assertEqual(d['weight'],.025);self.assertEqual(max(d['weighted_median_norm_ratio'].values()),.1)
        with self.assertRaises(ValueError):decide(rows[:11])
        rows[0]['HAnchor_norm']=float('nan')
        with self.assertRaises(ValueError):decide(rows)


if __name__=='__main__':unittest.main()
