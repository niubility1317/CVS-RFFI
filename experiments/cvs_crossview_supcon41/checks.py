"""Focused view-membership,default-head/data/budget criteria negatives."""
import unittest,copy
import torch
from .relations import penalty,select_maps,metadata,LEVELS
from .criteria import decide
from .protocol import loader_kind
class Checks(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7);self.a={k:torch.randn(128,4,8,requires_grad=True) for k in LEVELS};self.b={k:torch.randn(128,4,8,requires_grad=True) for k in LEVELS};self.y=torch.arange(128)%6;self.rx=torch.tensor((1,3,4,6,8))[torch.arange(128)%5];self.mask=torch.arange(128)%2==0
 def test_existing_as_graph_only_SAT(self):
  for mode in ('ORD128','DUP192','SAT192'):
   v,stats=penalty(self.a,self.b,self.y,self.rx,self.mask,mode);g=torch.autograd.grad(v,tuple(self.b.values()),allow_unused=True,retain_graph=True);self.assertEqual(any(x is not None and float(x.abs().sum())>0 for x in g),mode=='SAT192');self.assertTrue(all(s['same_RX_positive_pairs']==0 for s in stats))
 def test_duplicate_preserves_exact_ids(self):
  yy,rx=metadata(self.y,self.rx,self.mask,'DUP192');self.assertTrue(torch.equal(yy[128:],self.y[self.mask]));self.assertTrue(torch.equal(rx[128:],self.rx[self.mask]));self.assertEqual(len(yy),192)
 def test_view_membership_exact(self):
  for mode in ('ORD128','DUP192','SAT192'):
   maps=select_maps(self.a,self.b,self.mask,mode)
   for k,h in maps.items():
    self.assertTrue(torch.equal(h[:128],self.a[k]))
    if mode!='ORD128':self.assertTrue(torch.equal(h[128:],(self.a if mode=='DUP192' else self.b)[k][self.mask]))
 def test_reject_hidden_label_receiver_mask(self):
  for y,rx,mask,mode in [(self.y-6,self.rx,self.mask,'SAT192'),(self.y,self.rx*0,self.mask,'SAT192'),(self.y,self.rx,~(torch.arange(128)>1),'SAT192'),(self.y,self.rx,self.mask,'UAS')]:
   with self.assertRaises(ValueError):metadata(y,rx,mask,mode)
 def test_finite_zero_no_eligible_positives(self):
  for mode in ('ORD128','DUP192','SAT192'):
   v,s=penalty(self.a,self.b,self.y,torch.ones(128,dtype=torch.long),self.mask,mode);self.assertEqual(float(v),0.);self.assertTrue(all(x['valid_anchors']==0 for x in s))
 def test_no_nonfinite_drop(self):
  bad={k:v.detach().clone() for k,v in self.a.items()};bad[LEVELS[0]][0,0,0]=float('nan')
  with self.assertRaises(ValueError):penalty(bad,self.b,self.y,self.rx,self.mask,'DUP192')
 def test_decision_requires_both_controls_all_draws(self):
  c=dict(ORD128=.98,DUP192=.98,SAT192=.98);u={k:[[.8,.7,.6],[.8,.7,.6]] for k in c};u['SAT192']=[[.8,.71,.61],[.8,.71,.61]];self.assertTrue(decide(c,u)['source_qualified']);u['SAT192'][1][2]=.609;self.assertFalse(decide(c,u)['source_qualified']);u['SAT192'][1][2]=.61;c['SAT192']=.9774;self.assertFalse(decide(c,u)['source_qualified'])
 def test_exact_own_checkpoint_loader(self):
  self.assertEqual(loader_kind('/own20/final_ssdg.pth','/own20/final_ssdg.pth','/own30/final_ssdg.pth'),'parent')
  with self.assertRaises(ValueError):loader_kind('/unverified/final_ssdg.pth','/own20/final_ssdg.pth','/own30/final_ssdg.pth')
if __name__=='__main__':unittest.main()
