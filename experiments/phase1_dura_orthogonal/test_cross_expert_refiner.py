import pytest
import torch

from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner


def make_inputs():
    torch.manual_seed(7)
    x=torch.randn(3,2,256)
    anchor=torch.randn_like(x)
    direction=torch.randn_like(x)
    proposal=torch.randn_like(x)
    variance=torch.full((3,),.03)
    return x,anchor,direction,proposal,variance


def test_initial_prediction_is_half_mix_and_input_validation():
    model=CrossExpertRefiner(width=32).eval()
    x,a,d,p,v=make_inputs()
    torch.testing.assert_close(model(x,a,d,p,v),(a+p)*.5,atol=5e-7,rtol=1e-6)
    with pytest.raises(ValueError):
        model(x,a,d,p,v[:,None])


def test_unrestricted_correction_and_both_branches_receive_gradient():
    model=CrossExpertRefiner(width=32).train()
    x,a,d,p,v=make_inputs()
    out=model(x,a,d,p,v)
    loss=(out-x).square().mean()
    loss.backward()
    for parameter in (model.time_out.weight,model.freq_out.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    with torch.no_grad():
        model.time_out.bias.fill_(2.)
        corrected=model(x,a,d,p,v)
    assert torch.isfinite(corrected).all()
    assert ((corrected-(a+p)*.5).abs()>1.).any()
