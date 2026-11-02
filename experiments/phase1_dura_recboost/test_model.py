import math
import unittest
import torch
from experiments.phase1_dura_context.model import ContextRecovery
from experiments.phase1_dura_recboost.model import WideContextRecovery
from experiments.phase1_dura_recboost.objective import recognition_source_objective
from experiments.phase1_volterra_aligned.objective import source_loss

def rotate(x,angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),dim=1)

class RecognitionBoostTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(492)
        self.x=torch.randn(3,2,256);self.anchor=torch.randn(3,2,256);self.lv=torch.randn(3)
    def test_zero_output_equals_anchor(self):
        model=WideContextRecovery().eval()
        torch.testing.assert_close(model(self.x,self.anchor,self.lv),self.anchor,rtol=0,atol=0)
    def test_wide_residual_is_four_times_base(self):
        base=ContextRecovery().eval();wide=WideContextRecovery().eval()
        with torch.no_grad():
            base.gate_output.weight.normal_(0,.03);base.gate_output.bias.normal_(0,.03)
        wide.load_state_dict(base.state_dict())
        y=base(self.x,self.anchor,self.lv);z=wide(self.x,self.anchor,self.lv)
        self.assertGreater((y-self.anchor).abs().sum().item(),0)
        torch.testing.assert_close(z-self.anchor,4*(y-self.anchor),rtol=2e-5,atol=5e-7)
    def test_phase_equivariance(self):
        model=WideContextRecovery().eval()
        with torch.no_grad():
            model.gate_output.weight.normal_(0,.03);model.gate_output.bias.normal_(0,.03)
            y=model(self.x,self.anchor,self.lv)
            z=model(rotate(self.x,.59),rotate(self.anchor,.59),self.lv)
        torch.testing.assert_close(z,rotate(y,.59),rtol=2e-5,atol=2e-5)
    def test_pseudo_class_term_has_real_gradient(self):
        class Reader(torch.nn.Module):
            def features(self,x):return x.flatten(1)[:,:6]
            def classify_features(self,x):return x
        reader=Reader().eval();out=torch.randn(4,2,256,requires_grad=True);clean=torch.randn(4,2,256)
        features=reader.features(clean).detach();groups=torch.tensor([4,4,8,8])
        base,_=source_loss(reader,out,clean,features,groups)
        augmented,terms=recognition_source_objective(reader,out,out.detach(),clean,features,groups)
        extra=augmented-base
        self.assertGreater(extra.item(),0)
        self.assertGreater(torch.autograd.grad(extra,out)[0].abs().sum().item(),0)
        self.assertTrue(torch.isfinite(terms['pseudo_ce']).all())

if __name__=='__main__':unittest.main()
