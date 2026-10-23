from copy import deepcopy
import math
import unittest
import torch

from experiments.pamf_ssl_research.plan import configuration as base_config, SEEDS
from experiments.pamf_ssl_research.paired_u import configuration, validate
from experiments.pamf_ssl_research.paired_u_objective import paired_terms, PairedObjective
from experiments.pamf_ssl_research.objective import Objective
from tests.test_pamf_ssl_research import template


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature=torch.nn.Linear(2,160)
        self.head=torch.nn.Linear(160,6)
        with torch.no_grad(): self.head.bias[0]=15.
    def features(self,x): return self.feature(x.mean(-1))
    def classify_features(self,z): return self.head(z)


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone=torch.nn.Module()
        self.id_backbone.encoder=Encoder()
        self._cd_U_seen=self._cd_U_accepted=0
        self.physical_records=[]
    def _cd_practical(self,x,meta,*args):
        self.physical_records.append((x.detach().clone(),meta['physical_id'].clone()))
        return x+.3
    def forward_identity_only(self,x,domain_labels):
        self.last_inputs=x.detach().clone()
        self.last_domains=domain_labels.clone()
        z=self.id_backbone.encoder.features(x)
        self.last_features=z
        z.retain_grad()
        return dict(z_id=z,tx_logits=self.id_backbone.encoder.classify_features(z))


def setup():
    parent=base_config(template(),'RXPSM',SEEDS[0],'/p','old')
    c=configuration(parent,'/p','new')
    student=Student()
    x=torch.randn(8,2,256)
    meta=dict(rx_i=torch.tensor([1,3,4,6,8,1,3,4]),physical_id=torch.arange(8)*17+5)
    domain=torch.arange(8)
    student._cd_U_loader=[(x,torch.full((8,),-1),domain,meta)]
    student._cd_U_iter=iter(student._cd_U_loader)
    obj=PairedObjective(student,c)
    inherited=Objective(student,parent).state_dict()
    inherited.update(ema_updates=6660,seen=963900,physical_seen=481950,calls=3774)
    obj.load_state_dict(inherited)
    obj.current_epoch,obj.current_step=31,6661
    obj.labeled=(x,torch.zeros(8,dtype=torch.long),meta['rx_i'])
    return student,obj,c,x,meta,domain


class PairedUTest(unittest.TestCase):
    def test_literal_loss_coefficients_and_detached_teacher_target_weight(self):
        logits=torch.tensor([[0.,0.],[0.,math.log(3)]],requires_grad=True)
        features=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
        teacher=torch.tensor([[1.,0.]],requires_grad=True)
        target=torch.tensor([[1.,0.]],requires_grad=True)
        weight=torch.ones(1,requires_grad=True)
        terms=paired_terms(dict(tx_logits=logits,z_id=features),teacher,target,weight,None,torch.tensor([False]))
        loss=sum(terms.values())
        self.assertAlmostEqual(float(loss.detach()),.5*math.log(2)+.25*math.log(4)+.10,places=6)
        loss.backward()
        self.assertGreater(float(logits.grad[0].norm()),0.)
        self.assertGreater(float(logits.grad[1].norm()),0.)
        self.assertGreater(float(features.grad[1].norm()),0.)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(weight.grad)

    def test_both_prototype_terms_use_detached_visible_label_anchors(self):
        centers=torch.tensor([[[0.,1.],[1.,0.]]],requires_grad=True)
        features=torch.tensor([[1.,0.],[1.,0.]],requires_grad=True)
        terms=paired_terms(dict(tx_logits=torch.zeros(2,2,requires_grad=True),z_id=features),
            torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]]),torch.ones(1),centers,torch.tensor([True]))
        self.assertAlmostEqual(float(terms['clean_prototype'].detach()),.0125,places=7)
        self.assertAlmostEqual(float(terms['physical_prototype'].detach()),.0125,places=7)
        (terms['clean_prototype']+terms['physical_prototype']).backward()
        self.assertTrue(bool((features.grad.norm(dim=1)>0).all()))
        self.assertIsNone(centers.grad)

    def test_every_unique_record_has_ordered_clean_physical_pair_and_one_teacher_count(self):
        student,obj,c,x,meta,domain=setup()
        loss=obj.loss(student,'cpu')
        torch.testing.assert_close(student.last_inputs[:8],x)
        torch.testing.assert_close(student.last_inputs[8:],x+.3)
        torch.testing.assert_close(student.last_domains,domain.repeat(2))
        torch.testing.assert_close(student.physical_records[0][1],meta['physical_id'])
        self.assertEqual(len(student.physical_records),1)
        loss.backward()
        self.assertTrue(bool((student.last_features.grad.norm(dim=1)>0).all()))
        self.assertTrue(all(p.grad is None for p in obj.teacher.parameters()))
        self.assertEqual(obj.seen,963908)
        self.assertEqual(obj.physical_seen,481958)
        self.assertEqual(obj.student_U_views,963916)
        self.assertEqual(obj.paired_clean_seen,8)
        self.assertEqual(obj.paired_physical_seen,8)
        obj.after_update(student)
        saved=obj.state_dict()
        restored=PairedObjective(student,c)
        restored.load_state_dict(saved)
        self.assertEqual(restored.ema_updates,6661)
        self.assertEqual(restored.calls,3775)
        self.assertEqual(restored.paired_calls,1)
        self.assertEqual(restored.student_U_views,963916)

    def test_hidden_truth_and_different_recipe_are_rejected(self):
        student,obj,c,x,meta,domain=setup()
        student._cd_U_iter=iter([(x,torch.zeros(8,dtype=torch.long),domain,meta)])
        with self.assertRaisesRegex(ValueError,'truth exposed'):obj.loss(student,'cpu')
        self.assertEqual(student.physical_records,[])
        bad=deepcopy(c)
        bad['paired_u']['clean_CE']=.4
        with self.assertRaisesRegex(ValueError,'paired-U recipe'):validate(bad)
        bad=deepcopy(c)
        bad['tests']={'target':'forbidden'}
        with self.assertRaisesRegex(ValueError,'source handles'):validate(bad)

    def test_wrong_pair_shape_cannot_silently_omit_clean_rows(self):
        with self.assertRaisesRegex(ValueError,'aligned clean/physical'):
            paired_terms(dict(tx_logits=torch.zeros(1,6),z_id=torch.zeros(1,160)),
                torch.zeros(1,160),torch.ones(1,6)/6,torch.ones(1),None,torch.tensor([False]))


if __name__=='__main__':unittest.main()
