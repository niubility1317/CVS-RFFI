import unittest
import torch
from experiments.phase1_dura_iqproject.objective import projected_loss, projected_source_objective
from experiments.phase1_dura_safe_memory.objective import protected_source_objective

class IQProjectionTest(unittest.TestCase):
    def test_projection_only_changes_conflicting_samples(self):
        x=torch.tensor([[.1,.2],[.3,.4]],requires_grad=True)
        gw=torch.tensor([[1.,0.],[1.,0.]])
        gr=torch.tensor([[-2.,1.],[1.,1.]])
        wave=(x*gw).sum();recognition=(x*gr).sum()
        loss,info=projected_loss(wave,recognition,x)
        actual=torch.autograd.grad(loss,x)[0]
        torch.testing.assert_close(actual,torch.tensor([[1.,1.],[2.,1.]]))
        self.assertEqual(info['conflicts'],1)

    def test_zero_wave_gradient_remains_finite(self):
        x=torch.randn(2,2,requires_grad=True)
        wave=(x*0).sum();rec=(x*x).sum()
        loss,info=projected_loss(wave,rec,x)
        g=torch.autograd.grad(loss,x)[0]
        torch.testing.assert_close(g,2*x)
        self.assertEqual(info['conflicts'],0)

    def test_samplewise_projection_preserves_primary_direction(self):
        torch.manual_seed(39)
        x=torch.randn(7,2,9,requires_grad=True)
        gw=torch.randn_like(x);gr=torch.randn_like(x)
        loss,_=projected_loss((x*gw).sum(),(x*gr).sum(),x)
        got=torch.autograd.grad(loss,x)[0]
        rec_projected=(got-gw).flatten(1);wave=gw.flatten(1)
        self.assertTrue(torch.all((rec_projected*wave).sum(1)>=-1e-5))

    def test_requires_scalar_objectives(self):
        x=torch.randn(2,2,requires_grad=True)
        with self.assertRaises(ValueError):projected_loss(x,x.sum(),x)


    def test_forward_value_matches_protected_source_objective(self):
        class Reader(torch.nn.Module):
            def features(self, value):
                return value.flatten(1)[:,:6]
            def classify_features(self, value):
                return value
        reader=Reader().eval()
        torch.manual_seed(7)
        output=torch.randn(4,2,256,requires_grad=True)
        anchor=torch.randn(4,2,256)
        clean=torch.randn(4,2,256)
        features=reader.features(clean).detach()
        groups=torch.tensor([4,4,8,8])
        expected,_=protected_source_objective(reader,output,anchor,clean,features,groups)
        actual,_=projected_source_objective(reader,output,anchor,clean,features,groups)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main()
