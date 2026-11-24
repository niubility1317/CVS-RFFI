import torch

from experiments.pamf_lora_main.antithetic_detail import objective
from experiments.pamf_lora_main.carrier_coupled import observed_frame
from experiments.pamf_lora_main.channel_blind_source import FitRecipe
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.independent_shape_source import CALCULATE, NEW_ARM, RECIPES
from experiments.pamf_lora_main.relative_carrier import RelativeCarrierRecovery, mix


def test_source_relative_loss_does_not_penalize_retained_received_phase():
    tracker = PacketRestorer('consensus').eval()
    t = torch.arange(256)
    pattern = torch.polar(torch.ones_like(t,dtype=torch.float32),2*torch.pi*5*t/256)
    signal = (1+.05*pattern) * tracker.reference.chirp
    iq = real_iq(signal.expand(3,6,-1))

    def pair(clean,change,_perturb,_rng,*,with_identity):
        offset = torch.polar(torch.ones(6),torch.linspace(.35,1.3,6))[None,:,None]
        return torch.stack((clean,clean,clean+change,clean+change),dim=1)*offset[:,None]

    recipe = FitRecipe('identity',1,lambda:None,None,lambda model,y,q,r:(y,y))
    controls = dict(with_identity=False,recipe=recipe,step=1,split_risk=False,
                    build_inputs=pair,observation_frame=observed_frame,mix=mix)
    base,_ = objective(None,tracker,iq,None,**controls)
    corrected,terms = objective(None,tracker,iq,None,source_shape=True,**controls)
    assert base>.03
    assert corrected<base*.01
    assert terms.isfinite().all()


def test_complete_source_relative_training_retains_encoder_gradients():
    tracker = PacketRestorer('consensus').eval()
    rng = torch.Generator().manual_seed(433)
    u = torch.randn((6,6,256),generator=rng)
    v = torch.randn(u.shape,generator=rng)
    iq = real_iq(tracker.reference.chirp*(1+.02*torch.complex(u,v)))
    model = RelativeCarrierRecovery((4,8,8,8),coupled=True)
    loss,terms = CALCULATE(model,tracker,iq,rng,with_identity=True,recipe=RECIPES[NEW_ARM],step=1)
    loss.backward()
    assert terms.isfinite().all()
    for branch in (model.coarse,model.envelope):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads and all(g.isfinite().all() for g in grads)
        assert any(g.abs().sum()>0 for g in grads)
