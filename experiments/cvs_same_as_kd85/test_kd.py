import importlib.util,unittest
from pathlib import Path
import torch
import torch.nn.functional as F
s=importlib.util.spec_from_file_location('kd85',Path(__file__).with_name('kd.py'));kd=importlib.util.module_from_spec(s);s.loader.exec_module(kd)
class KDTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(85);self.a=torch.randn(64,6,dtype=torch.float64,requires_grad=True);self.b=torch.randn(64,6,dtype=torch.float64,requires_grad=True);self.teacher=torch.randn(64,6,dtype=torch.float64,requires_grad=True);self.y=self.teacher.detach().argmax(1)
 def test_native_teacher_shift_and_common_scale(self):
  a,use=kd.objective(self.a,self.b,self.teacher,self.y);q=self.teacher.detach().softmax(1);want=.5*(F.kl_div((self.a/6).log_softmax(1),q,reduction='batchmean')+F.kl_div((self.b/6).log_softmax(1),q,reduction='batchmean'));self.assertTrue(torch.allclose(a,want,atol=1e-12));self.assertTrue(bool(use.all()))
 def test_teacher_detached_student_joint_gradient(self):
  v,_=kd.objective(self.a,self.b,self.teacher,self.y);v.backward();self.assertIsNone(self.teacher.grad);self.assertGreater(float(self.a.grad.abs().sum()),0);self.assertGreater(float(self.b.grad.abs().sum()),0)
 def test_wrong_teacher_zero_and_finite(self):
  y=(self.y+1)%6;v,use=kd.objective(self.a,self.b,self.teacher,y);self.assertEqual(float(v),0);v.backward();self.assertFalse(bool(use.any()));self.assertEqual(float(self.a.grad.abs().sum()),0)
 def test_all_As_reduction_not_selected_denominator(self):
  y=self.y.clone();y[32:]=(y[32:]+1)%6;v,_=kd.objective(self.a,self.b,self.teacher,y);half,_=kd.objective(self.a[:32],self.b[:32],self.teacher[:32],self.y[:32]);self.assertTrue(torch.allclose(v,.5*half,atol=1e-12))
 def test_hidden_truth_rejected(self):
  y=self.y.clone();y[0]=-1
  with self.assertRaises(ValueError):kd.objective(self.a,self.b,self.teacher,y)
if __name__=='__main__':unittest.main()
