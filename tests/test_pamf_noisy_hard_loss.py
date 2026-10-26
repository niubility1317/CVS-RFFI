import unittest
import torch
from torch.nn import functional as F
from experiments.pamf_channel_pl.noisy_hard_loss import component,mixed_loss


class NoisyHardLossTests(unittest.TestCase):
    def test_GCE_value_bound_and_gradient_reweighting(self):
        z=torch.tensor([[1.,-.5,.2],[-4.,3.,1.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([0,0]);q=.7
        got=component(z,y,q);p=z.softmax(1).gather(1,y[:,None]).squeeze(1)
        self.assertTrue(torch.allclose(got,(1-p.pow(q))/q));self.assertTrue(bool(((got>=0)&(got<=1/q)).all()))
        g=torch.autograd.grad(got.sum(),z,retain_graph=True)[0];ce=F.cross_entropy(z,y,reduction='sum');gc=torch.autograd.grad(ce,z)[0]
        self.assertTrue(torch.allclose(g,p.detach().pow(q)[:,None]*gc))

    def test_CE_control_and_original_PL_fallback(self):
        z=torch.tensor([[1.,0.,-1.],[.2,.1,.4],[0.,2.,-.5]],requires_grad=True);pseudo=torch.tensor([0,2,1]);m=torch.tensor([True,False,True]);labels=torch.tensor([1,0,2]);b=torch.tensor([False,True,False])
        got=mixed_loss(z,pseudo,m,labels,b,q=0.,rho=0.)
        expected=(F.cross_entropy(z[:1],pseudo[:1])+F.cross_entropy(z[1:2],labels[1:2])+F.cross_entropy(z[2:],pseudo[2:]))/3
        self.assertTrue(torch.allclose(got,expected))
        zero=torch.zeros_like(b);fallback=mixed_loss(z,pseudo,m,labels,zero,q=.7,rho=.25)
        self.assertTrue(torch.allclose(fallback,(F.cross_entropy(z,pseudo,reduction='none')*m).mean()))

    def test_two_raw_views_preserve_noisy_loss_dose(self):
        z=torch.tensor([[1.,.2,-.5],[.4,.3,.7]],requires_grad=True);y=torch.tensor([2,0]);m=torch.ones(2,dtype=torch.bool);b=torch.tensor([True,False]);extra=z[b].detach().clone().requires_grad_()
        a=mixed_loss(z,y,m,y,b,q=.7,rho=.25,extra_logits=extra);c=mixed_loss(z,y,m,y,b,q=.7,rho=0.)
        self.assertTrue(torch.allclose(a,c));g,ge=torch.autograd.grad(a,(z,extra),retain_graph=True);gc=torch.autograd.grad(c,z)[0];g[b]+=ge
        self.assertTrue(torch.allclose(g,gc,atol=1e-7))

    def test_extreme_wrong_label_has_finite_bounded_loss(self):
        z=torch.tensor([[-10000.,10000.]],requires_grad=True);out=component(z,torch.tensor([0]),.7)
        self.assertTrue(torch.isfinite(out).all());out.sum().backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertLessEqual(float(out.detach()),1/.7+1e-6)

    def test_scaled_CE_control_uses_detached_mean_GCE_gradient_weight(self):
        z=torch.tensor([[2.,0.,-.5],[-1.,2.,.2]],dtype=torch.float64,requires_grad=True);y=torch.tensor([0,0]);m=torch.ones(2,dtype=torch.bool)
        got=mixed_loss(z,y,m,y,m,q=0.,rho=0.,uniform_attenuation=True)
        w=z.detach().softmax(1).gather(1,y[:,None]).pow(.7).mean()
        expected=w*F.cross_entropy(z,y)
        self.assertTrue(torch.allclose(got,expected));g=torch.autograd.grad(got,z,retain_graph=True)[0];ge=torch.autograd.grad(expected,z)[0]
        self.assertTrue(torch.allclose(g,ge))

    def test_invalid_dose_alignment_or_labels_rejected(self):
        z=torch.ones(2,3);y=torch.tensor([0,1]);m=torch.ones(2,dtype=torch.bool)
        for change in (dict(q=.3),dict(rho=.5),dict(noisy_mask=m.float()),dict(noisy_label=torch.tensor([0,3])),dict(rho=.25)):
            kw=dict(q=.7,rho=0.,noisy_label=y,noisy_mask=m);kw.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):mixed_loss(z,y,m,**kw)


if __name__=='__main__':unittest.main()
