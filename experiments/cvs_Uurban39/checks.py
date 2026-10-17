import unittest
import torch
from .rng import private_rng
from .protocol import loader_kind,supervision_weight
from .criteria import decide
from experiments.cvs_U_teacher38.soft_transfer import transfer_loss
class Checks(unittest.TestCase):
 def test_added_forward_does_not_advance_original_rng(self):
  torch.manual_seed(71);start=torch.get_rng_state().clone();x=torch.ones(8,4)
  with private_rng(x,2887):a=torch.nn.functional.dropout(x,p=.5,training=True)
  self.assertTrue(torch.equal(start,torch.get_rng_state()))
  with private_rng(x,2887):b=torch.nn.functional.dropout(x,p=.5,training=True)
  self.assertTrue(torch.equal(a,b))
  with private_rng(x,2888):c=torch.nn.functional.dropout(x,p=.5,training=True)
  self.assertFalse(torch.equal(a,c))
 @unittest.skipUnless(torch.cuda.is_available(),'No local CUDA runtime')
 def test_private_GPU_rng_restores_both_streams(self):
  x=torch.ones(8,4,device='cuda');cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
  with private_rng(x,2887):a=torch.nn.functional.dropout(x,p=.5,training=True)
  self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
  with private_rng(x,2887):b=torch.nn.functional.dropout(x,p=.5,training=True)
  self.assertTrue(torch.equal(a,b))
 def test_own_final_is_allowed_after_own_parent(self):
  self.assertEqual(loader_kind('/own/stage20/final.pth','/own/stage20/final.pth','/own/source/final.pth'),'parent');self.assertEqual(loader_kind('/own/source/final.pth','/own/stage20/final.pth','/own/source/final.pth'),'final')
  with self.assertRaises(ValueError):loader_kind('/other/model.pth','/own/stage20/final.pth','/own/source/final.pth')
 def test_zero_raw_and_urban_forward_controls(self):
  self.assertEqual(supervision_weight('B0',14,True),0);self.assertEqual(supervision_weight('UAS',14,False),0);self.assertEqual(supervision_weight('URAW',20,True),supervision_weight('UAS',20,True));self.assertEqual(supervision_weight('UAS',21,True),.16)
 def test_hidden_U_and_detached_teacher(self):
  t=torch.tensor([[12.,0,0,0,0,0]],requires_grad=True);s=torch.zeros(1,6,requires_grad=True)
  loss,_=transfer_loss(s,t,torch.tensor([-1]));loss.backward();self.assertIsNone(t.grad);self.assertGreater(float(s.grad.abs().sum()),0)
  with self.assertRaises(ValueError):transfer_loss(s,t,torch.tensor([0]))
 def test_source_criterion_rejects_missing_cells(self):
  c=dict(B0=.98,URAW=.98,UAS=.98);u=dict(B0=[[.8,.6,.5]]*2,URAW=[[.8,.6,.5]]*2,UAS=[[.8,.62,.52]]*2);t={k:True for k in c};self.assertTrue(decide(c,u,t)['source_qualified'])
  with self.assertRaises(ValueError):decide(c,dict(u,UAS=[]),t)
  u['UAS'][1]=[.8,.6,.52];self.assertFalse(decide(c,u,t)['source_qualified'])
 def test_clean_or_teacher_failure_cannot_promote(self):
  c=dict(B0=.98,URAW=.98,UAS=.97);u=dict(B0=[[.8,.6,.5]]*2,URAW=[[.8,.6,.5]]*2,UAS=[[.8,.7,.6]]*2);t={k:True for k in c};self.assertFalse(decide(c,u,t)['source_qualified']);c['UAS']=.98;t['UAS']=False;self.assertFalse(decide(c,u,t)['source_qualified'])
if __name__=='__main__':unittest.main()
