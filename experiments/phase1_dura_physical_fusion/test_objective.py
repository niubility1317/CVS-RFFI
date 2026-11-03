import unittest
from unittest.mock import patch
import torch
from torch import nn
from experiments.phase1_dura_physical_fusion.train_source import rescue_mask,fusion_training_objective
from experiments.phase1_dura_physical_fusion.model import PhysicalFusion

class Reader(nn.Module):
 def __init__(self):super().__init__();self.head=nn.Linear(512,6);self.eval().requires_grad_(False)
 def features(self,x):return x.flatten(1)
 def classify_features(self,x):return self.head(x)
class ObjectiveTest(unittest.TestCase):
 def test_mask_uses_only_confident_source_pseudo_repairs(self):
  clean=torch.tensor([[10.,0,0,0,0,0],[10.,0,0,0,0,0],[0.,0,0,0,0,0]])
  anchor=torch.tensor([[0.,10,0,0,0,0],[10.,0,0,0,0,0],[0.,10,0,0,0,0]])
  phys=torch.tensor([[10.,0,0,0,0,0],[10.,0,0,0,0,0],[10.,0,0,0,0,0]])
  mask,target=rescue_mask(clean,anchor,phys)
  self.assertEqual(mask.tolist(),[True,False,False]);self.assertEqual(target.tolist(),[0,0,0])
 def test_joint_objective_trains_model_only(self):
  torch.manual_seed(12);torch.set_num_threads(2);m=PhysicalFusion(width=8);reader=Reader()
  x=torch.randn(3,2,256);clean=torch.randn_like(x);packed=torch.cat((torch.randn_like(x),torch.randn_like(x)),1);lv=torch.full((3,),-2.)
  out=m(x,packed,lv);loss,terms=fusion_training_objective(reader,out,packed,clean,reader.features(clean),torch.tensor([4,4,8]),x,lv)
  loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(terms['physical_rescue']))
  self.assertGreater(float(m.gate_output.weight.grad.abs().sum()),0.)
  self.assertTrue(all(p.grad is None for p in reader.parameters()))

 def test_rescue_gradient_only_for_confident_pseudo_repair(self):
  class Direct(nn.Module):
   def features(self,x):return x[:,0,:6]
   def classify_features(self,x):return x
  anchor=torch.zeros(1,2,256);anchor[0,0,1]=10
  physical=torch.zeros_like(anchor);physical[0,0,0]=10
  clean_features=torch.tensor([[10.,0,0,0,0,0]])
  output=anchor.clone().requires_grad_();packed=torch.cat((anchor,physical),1)
  with patch('experiments.phase1_dura_physical_fusion.train_source.align_to_anchor',side_effect=lambda x,y,**k:x), \
       patch('experiments.phase1_dura_physical_fusion.train_source.protected_source_objective',side_effect=lambda reader,out,*a:(out.sum()*0,{})):
   loss,terms=fusion_training_objective(Direct(),output,packed,anchor,clean_features,torch.tensor([4]),anchor,torch.tensor([-2.]))
  loss.backward()
  self.assertEqual(terms['rescue_count'].item(),1);self.assertGreater(terms['physical_rescue'].item(),0)
  self.assertLess(output.grad[0,0,0].item(),0);self.assertGreater(output.grad[0,0,1].item(),0)

if __name__=='__main__':unittest.main()
