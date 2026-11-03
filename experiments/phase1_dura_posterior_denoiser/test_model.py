import unittest
import torch
from experiments.phase1_dura_posterior_denoiser.model import PosteriorDenoiser, denoising_loss


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((x[:,0]*c-x[:,1]*s, x[:,0]*s+x[:,1]*c), dim=1)


class PosteriorDenoiserTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20); torch.set_num_threads(2)
        self.x = torch.randn(2,2,256); self.anchor = torch.randn(2,2,256)
        self.logvar = torch.full((2,),-2.)
        self.model = PosteriorDenoiser(width=8).eval()

    def test_zero_correction_reconstructs_anchor_without_rng(self):
        torch.manual_seed(21); before=torch.random.get_rng_state().clone()
        with torch.no_grad(): out=self.model(self.x,self.anchor,self.logvar)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.allclose(out,self.anchor,atol=2e-6,rtol=1e-6))

    def test_learned_rollout_equivariant_and_batch_independent(self):
        with torch.no_grad():
            self.model.gate_output.weight.normal_(0,.03)
            self.model.gate_output.bias.normal_(0,.03)
            both=self.model(self.x,self.anchor,self.logvar)
            one=self.model(self.x[:1],self.anchor[:1],self.logvar[:1])
            angle=torch.tensor(.7)
            turned=self.model(rotate(self.x,angle),rotate(self.anchor,angle),self.logvar)
        self.assertTrue(torch.allclose(one,both[:1],atol=2e-5,rtol=2e-5))
        self.assertTrue(torch.allclose(turned,rotate(both,angle),atol=2e-5,rtol=2e-5))

    def test_denoising_objective_has_model_gradients_and_no_source_gradients(self):
        clean=self.anchor.clone().requires_grad_(); anchor=self.anchor.clone().requires_grad_()
        loss=denoising_loss(self.model,self.x,anchor,clean,self.logvar,torch.tensor([4,8]),
                            sigma=torch.tensor([.2,.3]),noise=torch.ones_like(self.x))
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(self.model.gate_output.weight.grad.abs().sum().item(),0.)
        self.assertIsNone(clean.grad);self.assertIsNone(anchor.grad)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError): self.model(self.x,self.anchor,self.logvar[:1])


if __name__=='__main__': unittest.main()
