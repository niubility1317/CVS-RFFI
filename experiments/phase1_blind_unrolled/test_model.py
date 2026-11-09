import unittest
import torch
from experiments.phase1_blind_unrolled.model import BlindUnrolledRecovery, forward_action, adjoint_action, TAPS

class BlindModelTest(unittest.TestCase):
    def test_real_linear_adjoint(self):
        torch.manual_seed(11)
        x=torch.randn(3,256,dtype=torch.complex64)
        y=torch.randn_like(x)
        a=torch.randn(3,TAPS,dtype=torch.complex64)*.03
        b=torch.randn_like(a)*.03
        lhs=(forward_action(x,a,b).conj()*y).sum(-1).real
        rhs=(x.conj()*adjoint_action(y,a,b)).sum(-1).real
        torch.testing.assert_close(lhs,rhs,rtol=1e-5,atol=1e-5)

    def test_identity_and_real_gradient(self):
        torch.manual_seed(12)
        model=BlindUnrolledRecovery(width=16,steps=2)
        received=torch.randn(2,2,256)
        output,reproject=model(received,return_physics=True)
        torch.testing.assert_close(output,received,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(reproject,received,rtol=1e-5,atol=1e-6)
        output.square().mean().backward()
        self.assertGreater(model.priors[0][-1].weight.grad.abs().sum().item(),0)

    def test_channel_head_receives_training_gradient(self):
        torch.manual_seed(14)
        model=BlindUnrolledRecovery(width=16,steps=2)
        received=torch.randn(2,2,256)
        clean=received+.2*torch.randn_like(received)
        for _ in range(2):
            restored,projected=model(received,return_physics=True)
            loss=(restored-clean).square().mean()+.02*(projected-received).square().mean()
            model.zero_grad(set_to_none=True);loss.backward()
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None: parameter.add_(parameter.grad,alpha=-.001)
        self.assertIsNotNone(model.channel_head.weight.grad)
        self.assertTrue(torch.isfinite(model.channel_head.weight.grad).all())
        self.assertGreater(model.channel_head.weight.grad.abs().sum().item(),0)

    def test_batch_independent(self):
        torch.manual_seed(13)
        model=BlindUnrolledRecovery(width=16,steps=2)
        with torch.no_grad():
            model.priors[0][-1].weight.normal_(0,.01)
            model.channel_head.weight.normal_(0,.01)
        received=torch.randn(3,2,256)
        full=model(received)
        parts=torch.cat([model(received[i:i+1]) for i in range(3)])
        torch.testing.assert_close(full,parts,rtol=1e-5,atol=2e-6)

if __name__ == '__main__': unittest.main()
