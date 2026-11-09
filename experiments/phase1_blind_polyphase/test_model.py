import unittest
import torch
from experiments.phase1_blind_unrolled.model import BlindUnrolledRecovery
from experiments.phase1_blind_polyphase.model import BlindPolyphaseRecovery, PolyphaseConv1d

class BlindPolyphaseTest(unittest.TestCase):
    def test_matching_output_and_gradients(self):
        torch.manual_seed(23)
        a=BlindUnrolledRecovery(width=16,steps=2)
        for prior in a.priors: torch.nn.init.normal_(prior[-1].weight,std=.01)
        b=BlindPolyphaseRecovery(width=16,steps=2)
        b.load_state_dict(a.state_dict(),strict=True)
        self.assertGreater(sum(isinstance(m,PolyphaseConv1d) for m in b.modules()),0)
        x=torch.randn(3,2,256);target=torch.randn_like(x)
        outputs=[]
        for m in (a,b):
            y,p=m(x,return_physics=True);((y-target).square().mean()+.02*(p-x).square().mean()).backward()
            outputs.append((y,p))
        for x,y in zip(*outputs):torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6)
        for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)

if __name__=='__main__':unittest.main()
