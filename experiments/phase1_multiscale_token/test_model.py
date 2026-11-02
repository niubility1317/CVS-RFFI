import importlib
import unittest
import torch


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1], s*x[:,0]+c*x[:,1]), dim=1)


class MultiScaleTokenTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(331)
        self.x = torch.randn(3,2,256)
        self.lv = torch.full((3,), -2.0)

    def model(self):
        try:
            m = importlib.import_module('experiments.phase1_multiscale_token.model')
        except ModuleNotFoundError as exc:
            if exc.name == 'experiments.phase1_multiscale_token.model':
                self.fail('independent multiscale network is not implemented')
            raise
        return m.MultiScaleToken()

    def test_initial_network_preserves_single_input_waveform(self):
        net = self.model()
        self.assertTrue(torch.equal(net(self.x,self.x,self.lv),self.x))

    def test_nonzero_network_respects_phase_and_batch_independence(self):
        net = self.model()
        torch.nn.init.normal_(net.gate_output.weight,std=.08)
        torch.nn.init.normal_(net.gate_output.bias,std=.08)
        angle = torch.tensor(.81)
        y = net(self.x,self.x,self.lv)
        torch.testing.assert_close(net(rotate(self.x,angle),rotate(self.x,angle),self.lv),rotate(y,angle),rtol=1e-4,atol=1e-4)
        single = torch.cat([net(self.x[i:i+1],self.x[i:i+1],self.lv[i:i+1]) for i in range(3)])
        torch.testing.assert_close(single,y,rtol=5e-5,atol=5e-5)

    def test_two_updates_train_shared_spectral_features(self):
        net = self.model()
        optim = torch.optim.AdamW(net.parameters(),lr=.003)
        target = self.x + .1*torch.randn_like(self.x)
        for _ in range(2):
            optim.zero_grad(set_to_none=True)
            (net(self.x,self.x,self.lv)-target).square().mean().backward()
            self.assertGreater(float(net.gate_output.weight.grad.abs().sum()),0)
            optim.step()
        self.assertTrue(torch.isfinite(net.entry.weight.grad).all())
        self.assertGreater(float(net.entry.weight.grad.abs().sum()),0)

    def test_rejects_incompatible_noise_vector(self):
        with self.assertRaises(ValueError):
            self.model()(self.x,self.x,self.lv[:2])


if __name__ == '__main__': unittest.main()
