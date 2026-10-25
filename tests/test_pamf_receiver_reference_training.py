from copy import deepcopy
import unittest
from unittest.mock import patch
import torch
from experiments.pamf_channel_pl.receiver_reference_training import config,validate_config,Objective,VARIANTS


def cfg(method):
    original=dict(run_id='original',row_id='F0-ordinary_PL-s2026092701',model_seed=2026092701,num_classes=6,
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],source_cache='source',source_contract='roles',ablation='F21')
    case=dict(protocol='F0',model_seed=2026092701,row_id=original['row_id'],parent_source='parent/segment20',expected_source_contract='roles',source_config=original)
    with patch('experiments.pamf_channel_pl.receiver_reference_training.read',return_value=dict(config=original)):
        c=config(case,'project','commit',method)
    c['source_G']=dict(kind='original_F0_LU',roles='LU')
    return c


class SourceHoldoutTests(unittest.TestCase):
    def test_three_rows_preserve_original_budget_head_and_reject_target_or_recipe_changes(self):
        for method in VARIANTS:
            c=validate_config(cfg(method));self.assertEqual(c['epoch_offset'],30);self.assertEqual(c['segment_epochs'],20)
            self.assertTrue(c['default_head_joint']);self.assertEqual(c['receiver_reference']['held_extra_rx'],1)
            for key,value in [('target_inputs','query'),('model_seed',2026092702),('segment_epochs',19),('default_head_joint',False)]:
                bad=deepcopy(c);bad[key]=value
                with self.assertRaises(ValueError):validate_config(bad)
            bad=deepcopy(c);bad['receiver_reference']['reference_updates']=1
            with self.assertRaises(ValueError):validate_config(bad)

    def test_runtime_hidden_U_holdout_and_distinct_paired_views(self):
        class Student(torch.nn.Module):
            def __init__(self,batch):
                super().__init__();self.feature=torch.nn.Linear(160,160);self.head=torch.nn.Linear(160,6)
                self._cd_U_loader=[batch];self._cd_U_iter=iter([batch]);self._cd_U_seen=self._cd_U_accepted=0
            def forward_identity_only(self,x,domain_labels=None):
                z=self.feature(x);return dict(z_id=z,tx_logits=self.head(z))
            def forward(self,x):return self.forward_identity_only(x)['tx_logits']
        class Cache:
            def __init__(self):self.calls=[]
            def take(self,metadata,k,device,with_IQ):
                self.calls.append((k,with_IQ));return torch.ones(4,6)/6,torch.ones(4,dtype=torch.bool),torch.ones(4,160)*(k+1) if with_IQ else None
        meta=dict(rx_i=torch.tensor([1,1,3,8]),tx_label_visible=torch.zeros(4,dtype=torch.bool))
        batch=(torch.randn(4,160),torch.full((4,),-1),torch.zeros(4,dtype=torch.long),meta)
        for method in VARIANTS:
            o=Objective.__new__(Objective);o.c=cfg(method);o.cache=Cache();o.references=torch.nn.functional.normalize(torch.randn(6,5,160),dim=2)
            o.calls=o.seen=o.raw_accepted=o.extra_accepted=o.held_seen=0;o.metrics={};student=Student(batch)
            value=o.loss(student,'cpu');value.backward()
            self.assertEqual(o.extra_accepted,2);self.assertEqual(o.held_seen,2);self.assertEqual(o.seen,4)
            self.assertEqual(o.cache.calls,[(0,False)] if method=='RawReference' else [(0,True)] if method=='RestoredReference' else [(0,True),(1,True)])
            self.assertTrue(torch.isfinite(student.feature.weight.grad).all())
            meta['tx_label_visible'][0]=True;student._cd_U_iter=iter([batch])
            with self.assertRaises(ValueError):o.loss(student,'cpu')
            meta['tx_label_visible'][0]=False


if __name__=='__main__':unittest.main()
