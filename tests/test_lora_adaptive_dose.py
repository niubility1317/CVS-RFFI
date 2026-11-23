import torch

from experiments.pamf_lora_main.adaptive_dose import AdaptiveDoseDenoise, dose_limit, dose_risk


def test_observation_conditioned_cap_and_noncarrier_support():
    sigma = torch.tensor([.1,.4,.6,1.2])
    torch.testing.assert_close(dose_limit(sigma,True),torch.tensor([.45,.46,.69,.98]))
    torch.testing.assert_close(dose_limit(sigma,False),torch.full_like(sigma,.45))

    torch.manual_seed(555)
    y = torch.randn(2,6,256,dtype=torch.complex64)
    q = .8*torch.ones(2,6,1)
    fixed = AdaptiveDoseDenoise(adaptive=False)
    adaptive = AdaptiveDoseDenoise(adaptive=True)
    adaptive.load_state_dict(fixed.state_dict())
    assert torch.equal(fixed(y,q),adaptive(y,q))
    assert torch.count_nonzero(fixed(y,q))==0
    with torch.no_grad():
        fixed.spectral.out.bias[0]=14.
        adaptive.load_state_dict(fixed.state_dict())
    u,sigma,fixed_cap = fixed(y,q,details=True)
    v,v_sigma,adaptive_cap = adaptive(y,q,details=True)
    torch.testing.assert_close(sigma,v_sigma)
    torch.testing.assert_close(v.mean(-1),torch.zeros_like(v[...,0]),atol=1e-6,rtol=0.)
    assert (u.abs().square().mean(-1).sqrt()<=fixed_cap[...,0]+1e-6).all()
    assert (v.abs().square().mean(-1).sqrt()<=adaptive_cap[...,0]+1e-6).all()
    assert (v-u).abs().max()>.01
    ordinary,tail = dose_risk(v,y,.75*y,sigma)
    assert torch.isfinite(torch.stack((ordinary,tail))).all()
    assert torch.autograd.grad(tail,adaptive.temporal.out.weight)[0].norm()>0
