import torch

from experiments.pamf_lora_main.pilot_joint import PilotJoint, estimate_pilot, pilot_inverse


def test_receive_pilot_fit_uses_the_correct_delay_sign():
    torch.manual_seed(104)
    phase = 2.*torch.pi*torch.rand(2,6,256)
    prefix = torch.polar(torch.ones_like(phase),phase)
    observed = (prefix+.18j*prefix.roll(1,-1))*prefix.conj()
    variance = torch.zeros(2,6,1)
    estimated = estimate_pilot(observed,prefix,variance)
    freq = torch.fft.fftfreq(256)
    desired = 1.+.18j*torch.polar(torch.ones_like(freq),-2.*torch.pi*freq)
    torch.testing.assert_close(estimated,desired.expand_as(estimated),atol=.006,rtol=.006)
    corrected = pilot_inverse(observed,observed,prefix,variance)
    torch.testing.assert_close(corrected.mean(-1),observed.mean(-1))
    assert (corrected-1.).abs().square().mean() < .1*(observed-1.).abs().square().mean()


def test_both_stages_train_together_with_finite_gradients():
    torch.manual_seed(55)
    wave = torch.randn(2,6,256,dtype=torch.complex64)
    phase = torch.rand(2,6,256)
    template = torch.polar(torch.ones_like(phase),phase)
    model = PilotJoint(equalize=True)
    q = .65*torch.ones(2,6,1)
    coarse,refined = model(wave,q,template)
    (-1.+coarse).abs().square().mean().add((refined-1.).abs().square().mean()).backward()
    for parameter in (model.coarse.spectral.out.weight,model.coarse.temporal.out.weight,model.prior.out.weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm()>0
