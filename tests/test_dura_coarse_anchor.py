import pytest
import torch
from experiments.pamf_dura_efficiency.coarse_anchor import coarse_anchor,ALPHA,CALIBRATION_RATIO


def test_clean_target_is_detached_and_packet_energy_is_separate():
    clean=torch.stack([torch.ones(2,256),torch.full((2,256),2.)]).requires_grad_()
    output=(clean.detach()+1).requires_grad_()
    guide,error=coarse_anchor(output,clean)
    torch.testing.assert_close(error,torch.tensor([1.,.25]))
    torch.testing.assert_close(guide,ALPHA*torch.tensor([1.,.25]))
    go,gx=torch.autograd.grad(guide.sum(),[output,clean],allow_unused=True)
    assert gx is None and go is not None and torch.isfinite(go).all()
    assert ALPHA*CALIBRATION_RATIO==pytest.approx(.25)


def test_zero_energy_floor_and_invalid_shape_are_explicit():
    clean=torch.zeros(2,2,256);output=torch.ones_like(clean,requires_grad=True)
    guide,error=coarse_anchor(output,clean)
    torch.testing.assert_close(error,torch.full((2,),10000.))
    assert torch.isfinite(torch.autograd.grad(guide.mean(),output)[0]).all()
    for bad in [clean[:,:,:128],clean[:,0],clean[0]]:
        with pytest.raises(ValueError):coarse_anchor(bad,clean)


def test_auxiliary_gradient_only_follows_existing_coarse_path():
    coarse=torch.nn.Parameter(torch.ones(2,256));action=torch.nn.Parameter(torch.ones(2,256));prior=torch.nn.Parameter(torch.ones(2,256))
    output=coarse[None].expand(3,-1,-1);clean=torch.full((3,2,256),.5)
    guide,_=coarse_anchor(output,clean)
    gradients=torch.autograd.grad(guide.mean(),[coarse,action,prior],allow_unused=True)
    assert gradients[0] is not None and gradients[0].abs().sum()>0
    assert gradients[1] is None and gradients[2] is None
