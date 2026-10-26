from contextlib import contextmanager
from types import SimpleNamespace,MethodType
from unittest.mock import patch
import unittest
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.labeled_h_anchor import anchored_contrast,LabeledHAnchor,anchor_gradient_norms
from experiments.pamf_channel_pl.clean_h_objective import CleanHObjective,physical_mask


class LabeledHAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_true_L_cross_RX_positives_only_and_whole_query_normalization(self):
        torch.manual_seed(3);q=torch.randn(3,160,dtype=torch.double,requires_grad=True);k=torch.randn(5,160,dtype=torch.double,requires_grad=True)
        pseudo=torch.tensor([0,0,1]);accepted=torch.tensor([True,False,True]);rx=torch.tensor([1,2,2]);y=torch.tensor([0,0,1,1,2]);krx=torch.tensor([1,2,2,3,1])
        loss,d=anchored_contrast(q,pseudo,accepted,rx,k,y,krx)
        scores=F.normalize(q,dim=1,eps=1e-4)@F.normalize(k.detach(),dim=1,eps=1e-4).T/.2
        expected=(torch.logsumexp(scores[0,[1,2,3,4]],0)-scores[0,1]+torch.logsumexp(scores[2,[0,1,3,4]],0)-scores[2,3])/3
        self.assertTrue(torch.allclose(loss,expected,atol=1e-12));self.assertEqual(d,dict(queries=3,accepted=2,valid=2,positive_pairs=2,negative_pairs=6))
        a,b=torch.autograd.grad(loss,(q,k),allow_unused=True);hand=torch.autograd.grad(expected,q)[0]
        self.assertTrue(torch.allclose(a,hand,atol=1e-12));self.assertEqual(float(a[1].abs().max()),0);self.assertIsNone(b)

    def test_no_accepted_or_no_cross_RX_positive_is_zero_not_a_fake_class_relation(self):
        for accepted,rx in [(torch.tensor([False,False]),torch.tensor([2,2])),(torch.tensor([True,True]),torch.tensor([1,1]))]:
            q=torch.randn(2,160,requires_grad=True);k=torch.randn(2,160,requires_grad=True)
            loss,d=anchored_contrast(q,torch.tensor([0,0]),accepted,rx,k,torch.tensor([0,1]),torch.tensor([1,1]));a,b=torch.autograd.grad(loss,(q,k),allow_unused=True)
            self.assertEqual(d['valid'],0);self.assertEqual(float(loss.detach()),0);self.assertEqual(float(a.abs().max()),0);self.assertIsNone(b)

    def test_invalid_class_shape_and_nonfinite_features_rejected(self):
        q=torch.ones(2,160);k=torch.ones(2,160);args=[torch.tensor([0,1]),torch.ones(2,dtype=torch.bool),torch.tensor([1,2]),k,torch.tensor([0,1]),torch.tensor([2,1])]
        with self.assertRaises(ValueError):anchored_contrast(q[:,:159],*args)
        bad=list(args);bad[4]=torch.tensor([0,6])
        with self.assertRaises(ValueError):anchored_contrast(q,*bad)
        q[0,0]=torch.nan
        with self.assertRaises(FloatingPointError):anchored_contrast(q,*args)

    def test_original_head_fusion_parameters_are_feature_parameters_not_class_weights(self):
        class Fusion(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.randn(6,160));self.base_norm=torch.nn.LayerNorm(160);self.pa_norm=torch.nn.LayerNorm(160);self.gain=torch.nn.Parameter(torch.tensor(.8))
            def forward(self,x):return self.base_norm(x)+self.gain*self.pa_norm(x.flip(-1))
        model=torch.nn.Module();model.sequence_pool=torch.nn.Linear(160,160,bias=False);model.cls_head=Fusion()
        q=model.cls_head(model.sequence_pool(torch.randn(4,160)));keys=torch.randn(8,160)
        loss,_=anchored_contrast(q,torch.arange(4),torch.ones(4,dtype=torch.bool),torch.tensor([1,2,1,2]),keys,torch.arange(4).repeat(2),torch.tensor([2,1,2,1,2,1,2,1]))
        named=list(model.named_parameters());gradients=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True);norms=anchor_gradient_norms(named,gradients)
        by_name=dict((name,g) for (name,_),g in zip(named,gradients))
        self.assertIsNone(by_name['cls_head.weight']);self.assertGreater(float(by_name['cls_head.base_norm.weight'].norm()),0);self.assertGreater(float(by_name['cls_head.pa_norm.weight'].norm()),0);self.assertGreater(float(by_name['cls_head.gain'].abs()),0)
        self.assertEqual(norms['classification_weight'],0);self.assertGreater(norms['signature'],0);self.assertGreater(norms['other_features'],0)

    def model(self):
        class Encoder:
            def features(self,x):return x+0.
        model=SimpleNamespace(id_backbone=SimpleNamespace(encoder=Encoder()),_feature_l_steps=1);calls=[]
        def forward(this,x,*args,**kwargs):calls.append(x);return dict(features=this.id_backbone.encoder.features(x))
        model.forward_identity_only=MethodType(forward,model);return model,calls

    def objective(self):
        obj=LabeledHAnchor.__new__(LabeledHAnchor);obj.L_keys=obj.L_y=obj.L_rx=obj.L_original=obj.H_queries=None
        obj.LA_L_calls=obj.LA_L_captures=obj.LA_L_packets=obj.LA_H_captures=0;obj.c=dict(step_offset=4440);return obj

    def test_capture_only_two_existing_raw_L_forwards_and_restore_on_error(self):
        obj=self.objective();model,calls=self.model();original=model.forward_identity_only
        kwargs=dict(labels=torch.arange(128)%6,extra=(None,dict(tx_label_visible=torch.ones(128,dtype=torch.bool),rx_i=torch.arange(128)%5)))
        @contextmanager
        def base(*args):yield
        with patch.object(CleanHObjective,'labeled',base):
            with obj.labeled(model,kwargs):
                for i in range(5):model.forward_identity_only(torch.full((128,160),float(i),requires_grad=True))
                self.assertEqual(len(obj.L_keys),2);self.assertTrue(all(not k.requires_grad for k in obj.L_keys));self.assertTrue(bool((obj.L_keys[0]==3).all()));self.assertTrue(bool((obj.L_keys[1]==4).all()))
                for _ in range(6):model.forward_identity_only(torch.zeros(256,160))
            self.assertEqual(len(calls),11);self.assertIs(model.forward_identity_only,original);self.assertIsNone(obj.L_keys);self.assertEqual(obj.LA_L_packets,256)
            with self.assertRaises(RuntimeError):
                with obj.labeled(model,kwargs):raise RuntimeError('probe')
            self.assertIs(model.forward_identity_only,original);self.assertIsNone(obj.L_keys)
        kwargs['extra'][1]['tx_label_visible'][0]=False
        with self.assertRaises(ValueError):
            with obj.labeled(model,kwargs):pass

    def test_only_third_fourth_existing_U_forwards_supply_physical_queries(self):
        obj=self.objective();model,calls=self.model();obj._joint_forward_index=0;obj.H_queries=[]
        pseudo=torch.arange(124)%6;accepted=torch.arange(124)>10;rx=torch.arange(124)%5
        def base(this,student,x,*args):this._joint_forward_index+=1;return student.forward_identity_only(x)
        with patch.object(CleanHObjective,'unlabeled_student_forward',base):
            for i in range(6):obj.unlabeled_student_forward(model,torch.full((124,160),float(i)),None,pseudo,accepted,dict(rx_i=rx))
        self.assertEqual(len(calls),6);self.assertEqual(obj.LA_H_captures,2);self.assertEqual(len(obj.H_queries),2)
        mask=physical_mask(124,4441,'cpu')
        for i,(q,p,a,r) in enumerate(obj.H_queries):
            self.assertEqual(q.shape,(62,160));self.assertTrue(bool((q==i+2).all()));self.assertTrue(torch.equal(p,pseudo[mask]));self.assertTrue(torch.equal(a,accepted[mask]));self.assertTrue(torch.equal(r,rx[mask]))


if __name__=='__main__':unittest.main()
