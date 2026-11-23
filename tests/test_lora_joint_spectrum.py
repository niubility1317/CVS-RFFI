import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.joint_spectrum import JointSpectrum, combine, objective, split
from experiments.pamf_lora_main.joint_spectrum_source import RECIPES, recover


def test_common_private_merge_and_matched_finetuning():
    rng = torch.Generator().manual_seed(17)
    y = .1*torch.randn((2,6,256),dtype=torch.complex64,generator=rng)+1.
    torch.testing.assert_close(combine(split(y)),y)
    q = torch.rand((2,6,1),generator=rng)
    level = .2 + torch.rand((2,6,1),generator=rng)
    model = JointSpectrum(exchange=True)
    coarse,fine,_ = model.stages(y,q,level)
    shadow,own,_ = model.stages(y,q,level,exchange=False)
    assert torch.equal(coarse,shadow)
    assert not torch.equal(fine,own)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in JointSpectrum(False).parameters())
    for output in (coarse,fine,own):
        torch.testing.assert_close(output.mean(-1),y.mean(-1),rtol=0.,atol=2e-7)
        assert ((output-y)*level).abs().square().mean((1,2)).sqrt().max()<.33


def test_guard_replay_and_gradient_on_source_like_iq():
    tracker = PacketRestorer('consensus').eval()
    iq = real_iq(torch.randn((4,6,256),dtype=torch.complex64))
    models = [recipe.model() for recipe in RECIPES.values()]
    guarded,active,_ = recover(iq,tracker,models,float('inf'),compare=True)
    assert active==0 and torch.equal(guarded,iq.expand_as(guarded))
    restored,active,details = recover(iq,tracker,models,-1.,compare=True)
    assert restored.shape==(8,*iq.shape) and active==len(iq)
    assert torch.isfinite(details).all() and torch.isfinite(restored).all()
    delta = restored[-2:] - iq
    torch.testing.assert_close(delta.square().sum((2,3,4))[0],delta.square().sum((2,3,4))[1],
                               atol=1e-6,rtol=1e-5)
    for arm,model in zip(RECIPES,models,strict=True):
        loss,values = objective(model,tracker,iq,torch.Generator().manual_seed(11),
                               with_identity=True,recipe=RECIPES[arm],step=1)
        loss.backward()
        assert torch.isfinite(values).all()
        assert all(torch.isfinite(h[-1].weight.grad).all() and h[-1].weight.grad.norm()>0
                   for h in (model.coarse,model.refine))
