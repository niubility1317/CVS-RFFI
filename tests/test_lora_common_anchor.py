import torch

from experiments.pamf_lora_main.common_anchor import CommonAnchor, constrained_step, objective
from experiments.pamf_lora_main.common_anchor_source import RECIPES
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.joint_spectrum_source import recover


def test_anchor_preserves_received_mean_and_solves_ridge_system():
    rng = torch.Generator().manual_seed(414)
    y = torch.randn((3,6,21),dtype=torch.complex64,generator=rng)
    prior = torch.randn(y.shape,dtype=y.dtype,generator=rng)
    gain = 1.+.2*torch.randn(y.shape,dtype=y.dtype,generator=rng)
    variance = .1+torch.rand(y.shape,generator=rng)
    penalty = .8
    for anchor in (True,False):
        change = constrained_step(y,prior,gain,variance,penalty,anchor=anchor)
        output = y+change
        gradient = gain.conj()*(gain*output-y)/variance+penalty*(output-prior)
        if anchor:
            torch.testing.assert_close(output.mean(1),y.mean(1),atol=2e-7,rtol=1e-6)
            torch.testing.assert_close(gradient,gradient.mean(1,keepdim=True).expand_as(gradient),atol=4e-6,rtol=1e-6)
        else:
            assert gradient.abs().max()<4e-6


def test_staged_geometry_and_source_gradients():
    rng = torch.Generator().manual_seed(44)
    y = torch.randn((3,6,256),dtype=torch.complex64,generator=rng)
    q = torch.rand((3,6,1),generator=rng)
    level = .15+torch.rand((3,6,1),generator=rng)
    model = CommonAnchor(True)
    coarse,final,_ = model.stages(y,q,level)
    unanchored = model.stages(y,q,level,anchor_common=False)[1]
    for wave in (coarse,final):
        torch.testing.assert_close(wave.mean(1),y.mean(1),atol=2e-7,rtol=1e-6)
        torch.testing.assert_close(wave.mean(-1),y.mean(-1),atol=2e-7,rtol=1e-6)
    assert not torch.equal(unanchored,final)
    iq = real_iq(y)
    tracker = PacketRestorer('consensus').eval()
    for name,recipe in RECIPES.items():
        arm = recipe.model()
        loss,values = objective(arm,tracker,iq,rng,with_identity=True,recipe=recipe,step=1)
        loss.backward()
        assert torch.isfinite(values).all()
        assert all(h[-1].weight.grad.norm()>0 for h in (arm.estimator,arm.coarse,arm.refine)), name
    with torch.no_grad():
        options,active,_ = recover(iq,tracker,(model,),float('inf'),compare=True,switch='anchor_common')
        assert not active and torch.equal(options,iq.expand_as(options))
