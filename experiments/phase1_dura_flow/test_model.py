import importlib
import unittest
import torch


def rotate(x, angle):
    c,s=torch.cos(angle),torch.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),dim=1)


class DuraFlowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(221)
        self.x=torch.randn(3,2,256)
        self.anchor=self.x+.1*torch.randn_like(self.x)
        self.clean=self.x+.3*torch.randn_like(self.x)
        self.lv=torch.full((3,),-2.0)
        self.groups=torch.tensor([4,4,6])

    def module(self):
        try:
            return importlib.import_module('experiments.phase1_dura_flow.model')
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_flow.model':
                self.fail('DURA-conditioned flow is not implemented')
            raise

    def test_zero_initialized_rollout_is_exact_anchor(self):
        m=self.module().DuraFlow()
        self.assertTrue(torch.equal(m(self.x,self.anchor,self.lv),self.anchor))

    def test_nonzero_rollout_is_phase_equivariant_and_batch_independent(self):
        m=self.module().DuraFlow()
        torch.nn.init.normal_(m.gate_output.weight,std=.02)
        torch.nn.init.normal_(m.gate_output.bias,std=.02)
        angle=torch.tensor(.73)
        y=m(self.x,self.anchor,self.lv)
        actual=m(rotate(self.x,angle),rotate(self.anchor,angle),self.lv)
        torch.testing.assert_close(actual,rotate(y,angle),rtol=6e-5,atol=6e-5)
        first=m(self.x[:1],self.anchor[:1],self.lv[:1])
        torch.testing.assert_close(first,y[:1],rtol=6e-5,atol=6e-5)

    def test_source_flow_matching_trains_velocity_without_target_gradients(self):
        mod=self.module();m=mod.DuraFlow()
        anchor=self.anchor.clone().requires_grad_();clean=self.clean.clone().requires_grad_()
        loss=mod.flow_matching_loss(m,self.x,anchor,clean,self.lv,self.groups,time=torch.tensor([.2,.5,.8]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(m.gate_output.weight.grad).all())
        self.assertGreater(float(m.gate_output.weight.grad.abs().sum()),0.)
        self.assertIsNone(anchor.grad);self.assertIsNone(clean.grad)

    def test_rejects_misaligned_group_shape(self):
        mod=self.module();m=mod.DuraFlow()
        with self.assertRaises(ValueError):
            mod.flow_matching_loss(m,self.x,self.anchor,self.clean,self.lv,self.groups[:2])


if __name__=='__main__':unittest.main()
