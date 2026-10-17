import unittest
import numpy as np
import torch
from . import design as d
from .maths import teacher_summary
from .soft_transfer import transfer_loss
class Checks(unittest.TestCase):
 def test_teacher_detached_and_all_uncertain_urban_students_learn(self):
  t=torch.tensor([[12.,0,0,0,0,0],[0,12.,0,0,0,0]],requires_grad=True);s=torch.zeros(2,6,requires_grad=True);loss,info=transfer_loss(s,t,torch.full((2,),-1,dtype=torch.long));loss.backward()
  self.assertIsNone(t.grad);self.assertTrue(bool((s.grad.abs().sum(1)>0).all()));self.assertTrue(bool((info['q']>0).all()))
 def test_hidden_U_negative(self):
  for y in [torch.tensor([0]),torch.tensor([-1.])]:
   with self.assertRaises(ValueError):transfer_loss(torch.zeros(1,6),torch.zeros(1,6),y)
 def test_batch_mean_keeps_confidence_mass(self):
  t=torch.tensor([[12.,0,0,0,0,0],[0.,0,0,0,0,0]]);s=torch.zeros(2,6)
  full,_=transfer_loss(s,t,torch.full((2,),-1,dtype=torch.long));one,_=transfer_loss(s[:1],t[:1],torch.tensor([-1]));self.assertAlmostEqual(float(full*2),float(one),places=6)
 def test_qualification_uses_source_teacher_only(self):
  n=1800;y=np.arange(n)%6;rx=np.repeat([1,3,4,6,8],360);t=np.full((n,6),-12.);t[np.arange(n),y]=12.;s=t.copy();sat=np.zeros((2,3,n,6));sat[:,:,:,0]=1.
  r,q=teacher_summary(t,s,sat,y,rx,d.RULE);self.assertTrue(r['source_qualified']);self.assertGreater(float(q.mean()),.99)
  bad=t.copy();bad[:72]=np.roll(bad[:72],1,axis=1);r,_=teacher_summary(bad,s,sat,y,rx,d.RULE);self.assertFalse(r['checks']['teacher_precision']);self.assertFalse(r['source_qualified'])
 def test_finite_shape_negative(self):
  with self.assertRaises(ValueError):transfer_loss(torch.zeros(1,5),torch.zeros(1,5),torch.tensor([-1]))
  with self.assertRaises(ValueError):transfer_loss(torch.tensor([[float('nan')]*6]),torch.zeros(1,6),torch.tensor([-1]))
 def test_checkpoint_contract_negative(self):
  row=d.rows()[0];expected=dict(role_ids=dict(L_s=list(range(6300)),U_s=list(range(56700)),V=list(range(27000))));init=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],target_contact=False,source_roles='EXACT_MATCH');a=dict(from_scratch=True,baseline_ckpt='',teacher_ckpt='',checkpoint_selection='final_only',epochs=20);done=dict(target_access=False,epoch=20,optimizer_steps=4440)
  self.assertTrue(d.validate_metadata(row,expected,expected,init,a,done))
  for wrong in [dict(a,teacher_ckpt='external'),dict(a,baseline_ckpt='old'),dict(a,freeze_backbone=True),dict(a,checkpoint_selection='best_target')]:
   with self.assertRaises(ValueError):d.validate_metadata(row,expected,expected,init,wrong,done)
  wrong=dict(role_ids=dict(expected['role_ids'],V=[99]));
  with self.assertRaises(ValueError):d.validate_metadata(row,wrong,expected,init,a,done)
if __name__=='__main__':unittest.main()
