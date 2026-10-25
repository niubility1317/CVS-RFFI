from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from experiments.cric_dura_comparison.io import read
from experiments.pamf_channel_pl.task_scaled_training import config,validate_config,Objective,VARIANTS,WEIGHT,prepare


def cfg(method):
    original=dict(run_id='original',row_id='F0-ordinary_PL-s2026092701',model_seed=2026092701,num_classes=6,
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='source',source_contract='roles',ablation='F21')
    case=dict(protocol='F0',model_seed=2026092701,row_id=original['row_id'],parent_source='parent/segment20',expected_source_contract='roles',source_config=original)
    with patch('experiments.pamf_channel_pl.task_scaled_training.read',return_value=dict(config=original)):
        c=config(case,'project','commit',method)
    c['source_G']=dict(kind='original_F0_LU',roles='LU')
    return c


class TaskScaledTrainingTests(unittest.TestCase):
    def test_protocol_budget_dose_and_default_head_are_fixed(self):
        for method in VARIANTS:
            c=validate_config(cfg(method));self.assertEqual(c['epoch_offset'],30);self.assertEqual(c['segment_epochs'],20)
            self.assertTrue(c['default_head_joint']);self.assertEqual(c['receiver_reference']['weight'],WEIGHT)
            for key,value in [('target_inputs','query'),('model_seed',2026092702),('segment_epochs',19),('default_head_joint',False)]:
                bad=deepcopy(c);bad[key]=value
                with self.assertRaises(ValueError):validate_config(bad)
            for key,value in [('reference_updates',1),('task_scale_max',33.),('weight',.1),('held_extra_rx',3)]:
                bad=deepcopy(c);bad['receiver_reference'][key]=value
                with self.assertRaises(ValueError):validate_config(bad)

    def test_prepare_smoke_points_to_a_generated_scaled_candidate(self):
        with TemporaryDirectory() as root:
            context=dict(cases=[dict(model_seed=2026092701)],G=dict(kind='original_F0_LU',roles='LU'))
            rt=dict(context='context',run_root=root,project='project',code_commit='commit',frozen_sources={},release_root='release',python='python')
            with patch('experiments.pamf_channel_pl.task_scaled_training.read',return_value=context), \
                    patch('experiments.pamf_channel_pl.task_scaled_training.validate_context'), \
                    patch('experiments.pamf_channel_pl.task_scaled_training.config',side_effect=lambda case,project,commit,method:cfg(method)):
                jobs=prepare(rt)
            self.assertEqual(len(jobs),4)
            smoke=read(Path(root)/'configs/smoke.json');target=Path(smoke['train_config'])
            self.assertTrue(target.is_file());self.assertEqual(read(target)['paper_method'],'GTaskScaled')
            self.assertEqual({j['id'] for j in jobs if j['kind']=='train_eval'},set(VARIANTS))
            self.assertTrue(all(j['entrypoint']=='experiments.pamf_channel_pl.task_scaled_training' for j in jobs))

    def test_hidden_U_holdout_same_two_views_scale_and_joint_backward(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.head=torch.nn.Linear(160,6)
            def classify_features(self,z):return self.head(z)
        class Student(torch.nn.Module):
            def __init__(self,batch):
                super().__init__();self.feature=torch.nn.Linear(160,160);self.id_backbone=torch.nn.Module()
                self.id_backbone.encoder=Encoder();self.inputs=[]
                self._cd_U_loader=[batch];self._cd_U_iter=iter([batch]);self._cd_U_seen=self._cd_U_accepted=0
            def forward_identity_only(self,x,domain_labels=None):
                self.inputs.append(x.detach().clone());z=self.feature(x)
                return dict(z_id=z,tx_logits=self.id_backbone.encoder.classify_features(z))
            def forward(self,x):return self.forward_identity_only(x)['tx_logits']
        class Cache:
            def __init__(self):self.calls=[]
            def take(self,metadata,k,device,with_IQ):
                self.calls.append((k,with_IQ));q=torch.nn.functional.one_hot(torch.tensor([0,1,2,3]),6).float()
                return q,torch.ones(4,dtype=torch.bool),torch.ones(4,160)*(k+1) if with_IQ else None
        torch.manual_seed(42)
        meta=dict(rx_i=torch.tensor([1,1,3,8]),tx_label_visible=torch.zeros(4,dtype=torch.bool))
        batch=(torch.randn(4,160),torch.full((4,),-1),torch.zeros(4,dtype=torch.long),meta)
        for method in VARIANTS:
            o=Objective.__new__(Objective);o.c=cfg(method);o.cache=Cache();o.references=torch.nn.functional.normalize(torch.randn(6,5,160),dim=2)
            o.calls=o.seen=o.raw_accepted=o.extra_accepted=o.held_seen=0;o.scale_sum=o.scale_cap_count=o.scale_max=0.;o.metrics={}
            o.expected_extra_per_epoch=2;student=Student(batch)
            value=o.loss(student,'cpu');value.backward()
            self.assertEqual(o.extra_accepted,2);self.assertEqual(o.held_seen,2);self.assertEqual(o.seen,4)
            self.assertEqual(o.cache.calls,[(0,method!='RawTaskScaled')])
            expected=batch[0] if method=='RawTaskScaled' else torch.ones(4,160)
            for actual in student.inputs[-2:]:self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.isfinite(student.feature.weight.grad).all());self.assertGreater(student.feature.weight.grad.norm(),0)
            state=o.state_dict();self.assertLessEqual(state['scale_sum'],32*2);self.assertLessEqual(state['scale_max'],32)
            self.assertEqual(state['extra_student_forwards'],8);self.assertFalse(state['new_head'])
            if method=='GUniform':self.assertEqual(state['scale_sum'],2)
            meta['tx_label_visible'][0]=True;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            meta['tx_label_visible'][0]=False
            batch[1][0]=0;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            batch[1][0]=-1


if __name__=='__main__':unittest.main()
