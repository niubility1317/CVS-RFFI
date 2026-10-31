"""Bounded complex input filtering and fixed CE/KD/PL arithmetic."""
import importlib
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.receiver_input_core')


def test_identity_16_parameters_and_direct_reflect_complex_equation():
    m=api();adapter=m.BoundedIQ();g=torch.Generator().manual_seed(41);x=torch.randn(2,2,256,generator=g)
    assert sum(p.numel() for p in adapter.parameters())==16 and set(adapter.state_dict())=={'theta'}
    assert torch.equal(adapter(x),x) and adapter.prior().item()==0.
    with torch.no_grad():adapter.theta.copy_(torch.linspace(-.8,.7,16))
    fir,gamma=adapter.coefficients();assert fir.shape==(7,2) and gamma.shape==(2,)
    signal=np.asarray(x.tolist());h=np.asarray(fir.detach().tolist());im=np.asarray(gamma.detach().tolist());z=signal[:,0]+1j*signal[:,1]
    padded=np.pad(z,((0,0),(3,3)),mode='reflect');expected=sum((h[j,0]+1j*h[j,1])*padded[:,j:j+256] for j in range(7))+(im[0]+1j*im[1])*z.conj()
    np.testing.assert_allclose(np.asarray(adapter(x).detach().tolist()),np.stack((expected.real,expected.imag),1),rtol=2e-6,atol=5e-7)
    residual=fir.detach().clone();residual[3,0]-=1
    assert float(residual.abs().max())<=.05 and float(gamma.detach().abs().max())<=.05
    torch.testing.assert_close(adapter.prior(),adapter.theta.tanh().square().mean())


def test_conjugate_image_and_input_parameter_gradients():
    m=api();adapter=m.BoundedIQ();x=torch.randn(2,2,256,requires_grad=True)
    with torch.no_grad():adapter.theta[14]=.6;adapter.theta[15]=-.4
    _,gamma=adapter.coefficients();expected=torch.stack((x[:,0]+gamma[0]*x[:,0]+gamma[1]*x[:,1],x[:,1]+gamma[1]*x[:,0]-gamma[0]*x[:,1]),1)
    torch.testing.assert_close(adapter(x),expected);adapter(x).square().mean().backward()
    assert x.grad is not None and x.grad.norm()>0 and adapter.theta.grad is not None and adapter.theta.grad.norm()>0


def test_source_and_target_loss_equations_keep_teachers_detached():
    m=api();adapter=m.BoundedIQ();logits=torch.randn(5,6,requires_grad=True);teacher=torch.randn(5,6,requires_grad=True);labels=torch.arange(5)
    out=m.source_loss(logits,labels,teacher,adapter);ce=F.cross_entropy(logits,labels);kd=4*F.kl_div(F.log_softmax(logits/2,1),F.softmax(teacher.detach()/2,1),reduction='batchmean')
    torch.testing.assert_close(out['total'],ce+kd+.1*adapter.prior());out['total'].backward();assert logits.grad.norm()>0 and teacher.grad is None
    q=torch.randn(4,6,requires_grad=True);z=torch.randn(4,160,requires_grad=True);tz=torch.randn(4,160,requires_grad=True);mask=torch.tensor([True,False,True,False]);p=torch.arange(4)
    target=m.target_loss(q,p,mask,z,tz);pl=(F.cross_entropy(q,p,reduction='none')*mask).mean();anchor=(z-tz.detach()).square().sum(1).mean()
    torch.testing.assert_close(target['total'],pl+10*anchor);target['total'].backward()
    assert tz.grad is None and z.grad.norm()>0 and torch.equal(q.grad[~mask],torch.zeros_like(q.grad[~mask]))


def test_bad_input_or_parameter_has_no_fallback(subtests):
    m=api();adapter=m.BoundedIQ()
    for defect in ('float64','shape','empty','nan','theta_nan','theta_dtype'):
        with subtests.test(defect=defect):
            model=m.BoundedIQ();x=torch.zeros(1,2,256)
            if defect=='float64':x=x.double()
            elif defect=='shape':x=x[:,:,:255]
            elif defect=='empty':x=x[:0]
            elif defect=='nan':x[0,0,0]=torch.nan
            elif defect=='theta_nan':
                with torch.no_grad():model.theta[0]=torch.nan
            else:model.double()
            with pytest.raises(ValueError):model(x)
