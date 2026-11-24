import torch

from experiments.pamf_lora_main import relative_carrier as m
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.relative_carrier_source import TRAIN_ARMS


def power(value):
    return value.abs().square().mean((-2,-1))


def test_all_packet_readouts_hold_input_power_and_zero_fallback():
    a = torch.tensor([[[1.+.2j,1.-.2j],[.5+.3j,.5-.3j]],[[0j,0j],[0j,0j]]])
    b = .3*a + .25j
    torch.testing.assert_close(power(m.mix(a,b,.5)),power(a))
    torch.testing.assert_close(power(m.match_packet_power(b,a)),power(a))
    assert torch.equal(m.mix(a,-a,.5),a)


def test_true_relative_reference_ignores_noisy_target_carrier_cancellation():
    source = torch.tensor([[[1.+.3j,1.-.3j],[.4+.2j,.4-.2j]]])
    measured = torch.stack((.4*source, 1.2*source, .8*source, source),dim=1)
    fake_label = source * torch.tensor([1,-1])
    loss,scores = m.amplitude_risk(measured.flatten(0,1),measured.flatten(0,1),
                                   fake_label,measured,fake_label,source_frame=source,received_half=measured)
    torch.testing.assert_close(scores,torch.zeros_like(scores),rtol=0,atol=1e-6)
    assert loss<1e-8


def test_head_only_redistributes_chirp_gain():
    model = m.RelativeCarrierRecovery((4,8,8,8),coupled=True)
    with torch.no_grad():
        model.envelope[0].weight.zero_()
        model.envelope[0].bias.zero_()
        model.envelope[0].weight[0,0] = 1.
        model.envelope[-1].weight[:,0] = 1.
        model.envelope[-1].bias.fill_(.12)
    signal = torch.ones((2,6,256),dtype=torch.complex64) * torch.linspace(.2,1.5,6)[None,:,None]
    coherence = torch.ones_like(signal.real[..., :1])
    for gain in model.gains(signal,coherence):
        torch.testing.assert_close(gain.log().mean(1),torch.zeros(2),atol=1e-6,rtol=0)
        assert gain.min()>=.25 and gain.max()<=4
        assert gain.min()<.99 and gain.max()>1.01


def test_live_training_path_finite_with_matching_packet_power():
    tracker = PacketRestorer('consensus').eval()
    rng = torch.Generator().manual_seed(784)
    shape = (6,6,256)
    detail = torch.complex(torch.randn(shape,generator=rng),torch.randn(shape,generator=rng))
    iq = real_iq(tracker.reference.chirp * (1+.08*detail))
    for name,linked in (('relative_independent',False),('relative_coupled',True)):
        model = m.RelativeCarrierRecovery((4,8,8,8),coupled=linked)
        loss, diagnostics = m.training_objective(model,tracker,iq,rng,with_identity=True,
                                                  recipe=TRAIN_ARMS[name],step=1)
        loss.backward()
        assert diagnostics.isfinite().all()
        assert model.envelope[-1].weight.grad.isfinite().all()
        assert model.envelope[-1].weight.grad.abs().sum()>0
