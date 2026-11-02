import math
import unittest
import torch
from experiments.phase1_dura_feedback.model import FeedbackRecovery,pack_context

def rotate(x,angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),dim=1)

class FeedbackTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(1251)
        self.x=torch.randn(3,2,256);self.anchor=torch.randn(3,2,256)
        self.features=torch.randn(3,160);self.logits=torch.randn(3,6);self.lv=torch.randn(3)
        self.packed=pack_context(self.anchor,self.features,self.logits)
    def test_pack_has_one_record_condition(self):
        self.assertEqual(self.packed.shape,(3,3,256))
        torch.testing.assert_close(self.packed[:,:2],self.anchor)
        torch.testing.assert_close(self.packed[:,2,:160],self.features)
        torch.testing.assert_close(self.packed[:,2,160:166],self.logits)
    def test_zero_init_returns_dura(self):
        model=FeedbackRecovery().eval()
        torch.testing.assert_close(model(self.x,self.packed,self.lv),self.anchor,rtol=0,atol=0)
    def test_condition_affects_nonzero_output(self):
        model=FeedbackRecovery().eval()
        with torch.no_grad():
            model.gate_output.weight.normal_(0,.04)
            first=model(self.x,self.packed,self.lv)
            other=pack_context(self.anchor,torch.randn_like(self.features),torch.randn_like(self.logits))
            second=model(self.x,other,self.lv)
        self.assertGreater((first-second).abs().sum().item(),1e-6)
    def test_phase_rotation_given_same_condition(self):
        model=FeedbackRecovery().eval()
        with torch.no_grad():
            model.gate_output.weight.normal_(0,.03);model.gate_output.bias.normal_(0,.03)
            original=model(self.x,self.packed,self.lv)
            rotated=pack_context(rotate(self.anchor,.42),self.features,self.logits)
            actual=model(rotate(self.x,.42),rotated,self.lv)
        torch.testing.assert_close(actual,rotate(original,.42),rtol=3e-5,atol=3e-5)
    def test_batch_independent(self):
        model=FeedbackRecovery().eval()
        with torch.no_grad():
            model.gate_output.weight.normal_(0,.03)
            all_=model(self.x,self.packed,self.lv)
            one=model(self.x[:1],self.packed[:1],self.lv[:1])
        torch.testing.assert_close(one,all_[:1],rtol=2e-5,atol=2e-5)
    def test_rejects_nonfinite_condition(self):
        self.packed[0,2,0]=float('nan')
        with self.assertRaises(FloatingPointError):
            FeedbackRecovery()(self.x,self.packed,self.lv)


    def test_condition_branch_learns_after_initial_step(self):
        model=FeedbackRecovery();opt=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad();loss=(model(self.x,self.packed,self.lv)-self.x).square().mean()
            loss.backward();opt.step()
        self.assertIsNotNone(model.condition[0].weight.grad)
        self.assertGreater(model.condition[0].weight.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()
