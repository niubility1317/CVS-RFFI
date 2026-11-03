import unittest
from types import SimpleNamespace
from unittest.mock import patch
import experiments
import torch
from experiments.phase1_iterative_anchor_fullsource.replay import model_for_schema,paired_outputs
from experiments.phase1_iterative_timefreq.model import IterativeTimeFreq
from experiments.phase1_dura_iterative_timefreq.model import IterativeDura

class FactoryTest(unittest.TestCase):
 def test_registered_factories_survive_path_reset(self):
  f=dict(first_factory=lambda:SimpleNamespace(kind='raw'),second_factory=lambda:SimpleNamespace(kind='dura'))
  with patch.object(experiments,'__path__',[]):
   self.assertEqual(model_for_schema('phase1_iterative_timefreq_v1',**f).kind,'raw');self.assertEqual(model_for_schema('phase1_dura_iterative_timefreq_v1',**f).kind,'dura')
  with self.assertRaises(ValueError):model_for_schema('invalid',**f)
 def test_actual_factories_create_registered_models(self):
  self.assertIs(type(model_for_schema('phase1_iterative_timefreq_v1')),IterativeTimeFreq);self.assertIs(type(model_for_schema('phase1_dura_iterative_timefreq_v1')),IterativeDura)
 def test_each_model_uses_its_own_anchor_and_closed_gate(self):
  x=torch.randn(3,2,256);anchor=torch.randn_like(x);active=torch.tensor([False,True,True]);calls=[]
  def blind(z):calls.append(len(z));return torch.zeros(len(z),4),torch.ones(len(z))
  def first(z,init,lv):self.assertTrue(torch.equal(z,init));return init+1
  def second(z,init,lv):self.assertTrue(torch.equal(init,anchor[active]));return init+2
  a,b=paired_outputs(x,anchor,active,[first,second],blind)
  self.assertEqual(calls,[2]);self.assertTrue(torch.equal(a[~active],x[~active]));self.assertTrue(torch.equal(b[~active],anchor[~active]));self.assertTrue(torch.equal(a[active],x[active]+1));self.assertTrue(torch.equal(b[active],anchor[active]+2))
  calls.clear();paired_outputs(x,anchor,torch.zeros(3,dtype=torch.bool),[first,second],blind);self.assertFalse(calls)

if __name__=='__main__':unittest.main()
