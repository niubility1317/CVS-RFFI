import math
import unittest
import torch
from experiments.phase1_spectral_coupling.model import SpectralCoupling


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)


class SpectralCouplingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(174); torch.set_num_threads(2)
        self.x = torch.randn(3, 2, 256); self.base = torch.randn_like(self.x); self.lv = torch.randn(3)

    def test_zero_model_preserves_base(self):
        self.assertTrue(torch.equal(SpectralCoupling()(self.x, self.base, self.lv), self.base))

    def test_phase_equivariance_and_batch_isolation(self):
        m = SpectralCoupling(); torch.nn.init.normal_(m.gate_output.weight, std=.1)
        torch.nn.init.normal_(m.gate_output.bias, std=.1); angle = torch.tensor(.47)
        out = m(self.x, self.base, self.lv)
        torch.testing.assert_close(m(rotate(self.x,angle),rotate(self.base,angle),self.lv),rotate(out,angle),rtol=4e-5,atol=4e-5)
        torch.testing.assert_close(m(self.x[1:2],self.base[1:2],self.lv[1:2]),out[1:2],rtol=3e-5,atol=3e-5)

    def test_correction_couples_distant_time_samples(self):
        m = SpectralCoupling(); torch.nn.init.normal_(m.gate_output.weight, std=.1)
        x = self.x.clone().requires_grad_(True)
        grad = torch.autograd.grad(m(x,self.base,self.lv)[0,0,0],x)[0][0,:,64:]
        self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()), 1e-5)

    def test_represents_one_bin_frequency_translation(self):
        model = SpectralCoupling()
        strength = math.atanh(math.sqrt(7)/3.)
        with torch.no_grad():
            model.gate_output.bias[3] = -strength
            model.gate_output.bias[4] = strength
        z = torch.complex(self.x[:, 0], self.x[:, 1])
        shifted = torch.fft.ifft(torch.roll(torch.fft.fft(z, norm='ortho'), 1, -1), norm='ortho')
        expected = torch.stack((shifted.real, shifted.imag), 1)
        torch.testing.assert_close(model(self.x, self.x, self.lv), expected, rtol=3e-5, atol=3e-5)

    def test_two_steps_reach_feature_network(self):
        m = SpectralCoupling(); opt = torch.optim.AdamW(m.parameters(), lr=.003)
        for _ in range(2):
            opt.zero_grad(set_to_none=True); (m(self.x,self.base,self.lv)-self.x).square().mean().backward()
            self.assertGreater(float(m.gate_output.weight.grad.abs().sum()), 0.); opt.step()
        self.assertTrue(torch.isfinite(m.entry.weight.grad).all())
        self.assertGreater(float(m.entry.weight.grad.abs().sum()), 0.)

    def test_wrong_shape_is_rejected(self):
        with self.assertRaises(ValueError): SpectralCoupling()(self.x,self.base,torch.zeros(2))


if __name__ == '__main__': unittest.main()
