import unittest
from types import SimpleNamespace
from unittest.mock import patch
import experiments
import torch
from experiments.phase1_local_teacher_fullsource.replay import model_for_schema,active_outputs

class FactoryTest(unittest.TestCase):
 def test_registered_factories_survive_path_reset(self):
  factories=dict(first_factory=lambda:SimpleNamespace(kind='narrow'),second_factory=lambda:SimpleNamespace(kind='wide'))
  with patch.object(experiments,'__path__',[]):
   self.assertEqual(model_for_schema('phase1_dura_physical_wide_v1',**factories).kind,'narrow')
   self.assertEqual(model_for_schema('phase1_dura_physical_local_teacher_v1',**factories).kind,'wide')
  with self.assertRaises(ValueError):model_for_schema('invalid',**factories)
 def test_actual_factories_reconstruct_same_wide_architecture(self):
  from experiments.phase1_dura_physical_wide.model import WidePhysicalFusion
  a=model_for_schema('phase1_dura_physical_wide_v1');b=model_for_schema('phase1_dura_physical_local_teacher_v1')
  self.assertIs(type(a),WidePhysicalFusion);self.assertIs(type(b),WidePhysicalFusion)
  self.assertEqual(set(a.state_dict()),set(b.state_dict()))
 def test_shared_physical_evidence_and_closed_gate(self):
  x=torch.randn(3,2,256);anchor=torch.randn_like(x);active=torch.tensor([False,True,True]);calls=[]
  def physical(z):calls.append(len(z));return z*.5,torch.ones(len(z))
  def first(z,packed,lv):
   self.assertTrue(torch.equal(packed[:,:2],anchor[active]));self.assertTrue(torch.equal(packed[:,2:],z*.5));return packed[:,:2]+1
  def second(z,packed,lv):return packed[:,:2]+2
  a,b=active_outputs(x,anchor,active,[first,second],physical)
  self.assertEqual(calls,[2]);self.assertTrue(torch.equal(a[~active],anchor[~active]));self.assertTrue(torch.equal(b[~active],anchor[~active]))
  self.assertTrue(torch.equal(a[active],anchor[active]+1));self.assertTrue(torch.equal(b[active],anchor[active]+2))
  calls.clear();active_outputs(x,anchor,torch.zeros(3,dtype=torch.bool),[first,second],physical);self.assertFalse(calls)

if __name__=='__main__':unittest.main()
