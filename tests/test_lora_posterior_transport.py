import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.posterior_transport import (
    MAX_RMS_FRACTION, initialize, objective, transport_strength,
)
from experiments.pamf_lora_main.posterior_transport_source import MODES, RECIPES, calibrate, recover
from experiments.pamf_lora_main.variance_transport import physical_norm


def test_gaussian_transport_recovers_marginal_scale_exactly():
    noise = torch.tensor([.01, .2, 2.], dtype=torch.float64)
    clean = torch.tensor([.03, .1, .8], dtype=noise.dtype)
    sensitivity = clean/(clean+noise)
    var = sensitivity*noise
    strength = transport_strength(var,noise)
    assert torch.allclose(1. + strength*(sensitivity - 1.), sensitivity.sqrt())


def test_train_both_posteriors_and_observed_recovery_paths():
    rng=torch.Generator().manual_seed(1307)
    tracker=PacketRestorer('consensus').eval()
    packet=tracker.reference.chirp.expand(5,6,-1)
    selected=real_iq(packet + .12*torch.randn(packet.shape,dtype=packet.dtype,generator=rng))
    models=[]
    for recipe in RECIPES.values():
        model=recipe.model()
        initialize(model,tracker,selected,selected)
        loss,risk=objective(model,tracker,selected,rng,with_identity=True,recipe=recipe,step=1)
        loss.backward()
        assert torch.isfinite(risk).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
        model.eval()
        assert calibrate(model,tracker,selected)[1]['variance_diagnostic_packets']==5
        models.append(model)

    with torch.no_grad():
        output,count,stats=recover(selected,tracker,models,0.,compare=True)
        assert count==len(selected) and output.shape==(len(MODES)-2,*selected.shape)
        assert torch.isfinite(stats).all()
        standalone, one, _=recover(selected[:1],tracker,models[-1:],0.)
        assert one==1 and torch.allclose(standalone[1],output[3,:1],atol=1e-5)
        clean, count, _=recover(selected,tracker,models,float('inf'),compare=True)
        assert count==0 and torch.equal(clean,selected.expand_as(clean))
        y,_,_,rms=coordinates(tracker,selected)
        context=models[-1].observe(y)
        stage=models[-1].estimate(y,rms,context)[1]
        edit=models[-1].readout(y,rms,stage,context[2])
        assert (physical_norm(edit,rms) <= (MAX_RMS_FRACTION + 1e-5)*physical_norm(y,rms)).all()
        assert edit.mean(-1).abs().max()<1e-5
