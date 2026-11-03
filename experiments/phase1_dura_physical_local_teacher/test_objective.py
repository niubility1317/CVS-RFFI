import unittest
import torch
from experiments.phase1_dura_physical_local_teacher.objective import local_projection_target, local_teacher_loss

class LocalProjectionTest(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7); torch.set_num_threads(2)
  self.anchor=torch.randn(3,2,256)
  self.delta=torch.randn_like(self.anchor)*.2
  self.physical=self.anchor+self.delta
 def test_exact_complex_mixture(self):
  d=torch.complex(self.delta[:,0],self.delta[:,1]); correction=torch.complex(torch.tensor(.4),torch.tensor(.2))*d
  clean=self.anchor+torch.stack((correction.real,correction.imag),1)
  result=local_projection_target(self.anchor,self.physical,clean)
  self.assertTrue(torch.allclose(result,clean,atol=2e-6,rtol=2e-6))
 def test_identical_experts_leave_anchor_and_are_finite(self):
  target=local_projection_target(self.anchor,self.anchor,torch.randn_like(self.anchor))
  self.assertTrue(torch.equal(target,self.anchor));self.assertTrue(torch.isfinite(target).all())
 def test_samples_are_independent_and_target_detached(self):
  clean=(self.anchor+.5*self.delta).requires_grad_();anchor=self.anchor.requires_grad_();physical=self.physical.requires_grad_()
  batch=local_projection_target(anchor,physical,clean)
  single=torch.cat([local_projection_target(anchor[i:i+1],physical[i:i+1],clean[i:i+1]) for i in range(3)])
  self.assertTrue(torch.allclose(batch,single,atol=1e-6));self.assertFalse(batch.requires_grad)
 def test_loss_provides_finite_correction_gradient_without_reference_gradient(self):
  clean=(self.anchor+.5*self.delta).requires_grad_();anchor=self.anchor.requires_grad_();physical=self.physical.requires_grad_();output=self.anchor.detach().clone().requires_grad_()
  loss=local_teacher_loss(output,anchor,physical,clean,torch.tensor([4,4,8]))
  loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(output.grad.abs().sum()),0)
  self.assertIsNone(anchor.grad);self.assertIsNone(physical.grad);self.assertIsNone(clean.grad)

if __name__=='__main__':unittest.main()
