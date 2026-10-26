from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_channel_pl import u_realization_training as train
from experiments.pamf_channel_pl.u_realization_objective import schedule,auxiliary,VARIANTS,WEIGHTS,ROUTES
from experiments.pamf_channel_pl.feature_task_training import extra as old_task
from experiments.pamf_channel_pl.restoration_bridge import extra as old_relation
from experiments.pamf_channel_pl.asymmetric_hg import extra as old_joint
build_config=train.config


def cfg(method):
    original=dict(run_id='original',row_id='F0-ordinary_PL-s2026092701',model_seed=2026092701,num_classes=6,
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='source',source_contract='roles',ablation='F21')
    case=dict(protocol='F0',model_seed=2026092701,row_id=original['row_id'],parent_source='parent/segment20',expected_source_contract='roles',source_config=original)
    with patch('experiments.pamf_channel_pl.u_realization_training.read',return_value=dict(config=original)):
        c=build_config(case,'project','commit',method)
    c['source_G']=dict(kind='original_F0_LU',roles='LU');return c


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__();self.feature=torch.nn.Linear(512,160);self.id_backbone=torch.nn.Module();self.id_backbone.encoder=torch.nn.Module()
        self.id_backbone.encoder.head=torch.nn.Linear(160,6,bias=False);self.id_backbone.encoder.classify_features=self.id_backbone.encoder.head;self.calls=0
    def forward_identity_only(self,x,domain_labels=None):
        self.calls+=1;z=self.feature(x.flatten(1));return dict(z_id=z,tx_logits=self.id_backbone.encoder.classify_features(z))


class URealizationTrainingTests(unittest.TestCase):
    def test_scientific_budget_recipe_and_permissions_are_fixed(self):
        for method in VARIANTS:
            c=train.validate_config(cfg(method));self.assertEqual(c['u_realizations']['weight'],WEIGHTS[method])
            for key,value in [('tests',{}),('target_inputs','query'),('model_seed',2026092702),('segment_epochs',21),
                              ('default_head_joint',False),('parent_source','other/segment30'),('own_EMA',True),('asymmetric_hg',{})]:
                bad=deepcopy(c);bad[key]=value
                with self.subTest(method=method,key=key),self.assertRaises(ValueError):train.validate_config(bad)
            bad=deepcopy(c);bad['u_realizations']['weight']*=2
            with self.assertRaises(ValueError):train.validate_config(bad)

    def test_realizations_use_equal_epochs_without_consuming_native_RNG(self):
        before=torch.random.get_rng_state().clone();seen=Counter()
        for call in range(1,4441):
            r,k=schedule(call);self.assertEqual(k,(6660+call-1)%3);seen[r]+=124 if call%222==0 else 256
        self.assertEqual(dict(seen),dict.fromkeys(range(4),283500));self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual([schedule(222*e+1)[0] for e in range(20)],list(range(4))*5)
        for call in (0,4441,True):
            with self.assertRaises(ValueError):schedule(call)

    def test_new_adapter_preserves_each_original_loss_and_feature_gradient(self):
        torch.manual_seed(42);q=torch.nn.functional.one_hot(torch.tensor([0,0,1,1,2,2,0,1]),6).float()
        rx=torch.tensor([3,4,3,4,3,4,1,1]);mask=rx!=1;domain=torch.zeros(8,dtype=torch.long)
        views=[torch.randn(8,2,256),torch.randn(8,2,256)]
        for method in VARIANTS:
            a=Student();b=deepcopy(a)
            value,detail=auxiliary(a,views,domain,q,mask,rx,method,check=True)
            if method=='GTaskMulti':expected=sum(old_task(b,v,domain,q,mask,True) for v in views)/2
            elif method in ('GGCLMulti','HGCLMulti'):expected=old_relation(b,views,domain,q,mask,rx,'bridge')[0]
            else:expected=old_joint(b,views,domain,q,mask,rx,'GJoint')[0]
            torch.testing.assert_close(value,expected);value.backward();expected.backward()
            self.assertEqual(a.calls,2);self.assertIsNone(a.id_backbone.encoder.head.weight.grad)
            torch.testing.assert_close(a.feature.weight.grad,b.feature.weight.grad);self.assertGreater(float(a.feature.weight.grad.norm()),0)
            self.assertEqual(detail['valid_anchors']==0,method=='GTaskMulti')

    def test_all_four_rows_follow_one_actual_fresh_checkpoint_smoke(self):
        with TemporaryDirectory() as folder:
            context=dict(cases=[dict(model_seed=2026092701)],G=dict(kind='original_F0_LU',roles='LU'))
            rt=dict(context='context',run_root=folder,project='project',code_commit='commit',frozen_sources={},release_root='release',python='python')
            with patch('experiments.pamf_channel_pl.u_realization_training.read',return_value=context), \
                    patch('experiments.pamf_channel_pl.u_realization_training.validate_context'), \
                    patch('experiments.pamf_channel_pl.u_realization_training.config',side_effect=lambda case,project,commit,method:cfg(method)):
                jobs=train.prepare(rt)
            self.assertEqual(len(jobs),5);self.assertEqual(jobs[0]['kind'],'smoke')
            self.assertTrue(all(j['dependencies']==[jobs[0]['completion']] for j in jobs[1:]))
            self.assertEqual([read(p)['paper_method'] for p in read(Path(folder)/'configs/smoke.json')['train_configs']],list(VARIANTS))


if __name__=='__main__':unittest.main()
