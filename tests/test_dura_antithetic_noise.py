import pytest
import torch
from experiments.pamf_dura_efficiency.antithetic_noise import augment


class Action:
    def __call__(self,x):return (1+.1j)*x+.05*x.conj()
    def adjoint(self,x):return (1-.1j)*x+.05*x.conj()


def examples():
    torch.manual_seed(111);x=torch.randn(4,2,256);action=Action();signal=action(torch.complex(x[:,0],x[:,1]))
    noise=.03*torch.complex(torch.randn(4,256),torch.randn(4,256));y=signal+noise
    return x,torch.stack((y.real,y.imag),1),action,torch.full((4,),.001),noise


def test_sign_aug_preserves_power_and_scales_action_variance_with_same_gain():
    clean,y,action,var,noise=examples();original=y.clone();mask=torch.tensor([True,False,True,False])
    output,scaled,v,info=augment(y,clean,action,var,mask);observed=torch.complex(output[:,0],output[:,1]);x=torch.complex(clean[:,0],clean[:,1])
    residual=observed-scaled(x);expected=noise*torch.where(mask,-1.,1.)[:,None]*info['gain'][:,None]
    torch.testing.assert_close(residual,expected,atol=5e-7,rtol=3e-5)
    torch.testing.assert_close(output.square().mean((1,2)),y.square().mean((1,2)),atol=1e-7,rtol=2e-6)
    torch.testing.assert_close(v,var*info['gain'].square());torch.testing.assert_close(info['effective'],mask.float())
    torch.testing.assert_close(output[~mask],y[~mask],atol=0,rtol=0);torch.testing.assert_close(y,original,atol=0,rtol=0)
    p=torch.complex(torch.randn(4,256),torch.randn(4,256));q=torch.complex(torch.randn(4,256),torch.randn(4,256))
    torch.testing.assert_close((scaled(p).conj()*q).real.sum(),(p.conj()*scaled.adjoint(q)).real.sum(),atol=1e-5,rtol=2e-6)


def test_clean_replay_residue_is_not_turned_into_noise_and_references_detach():
    x,y,action,var,_=examples();z=action(torch.complex(x[:,0],x[:,1]));y=torch.stack((z.real,z.imag),1).requires_grad_();x.requires_grad_()
    output,_,v,info=augment(y,x,action,var,torch.ones(4,dtype=torch.bool))
    torch.testing.assert_close(output,y,atol=0,rtol=0);assert not info['effective'].any() and not output.requires_grad
    torch.testing.assert_close(v,var,atol=0,rtol=0);assert x.grad is None and y.grad is None


def test_invalid_source_shapes_power_and_variance_fail_explicitly():
    x,y,action,var,_=examples();mask=torch.ones(4,dtype=torch.bool)
    with pytest.raises(ValueError):augment(y[:,:,:128],x[:,:,:128],action,var,mask)
    with pytest.raises(ValueError):augment(y,x,action,var*0,mask)
    with pytest.raises(ValueError):augment(y*0,x,action,var,mask)
    with pytest.raises(ValueError):augment(y,x,action,var,mask.float())
