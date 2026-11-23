import torch

from experiments.pamf_lora_main.complex_vector import CAP, ComplexVector, rms, teacher_vectors, vector_risk
from experiments.pamf_lora_main.complex_vector_source import MODES, readouts
from experiments.pamf_lora_main.denoise_components import bands, split_along
from experiments.pamf_lora_main.dura_repair import PacketRestorer


def test_vector_teachers_share_power_and_joint_teacher_balances_components():
    torch.manual_seed(42)
    desired = torch.randn(3, 6, 256, dtype=torch.complex64)
    reference = .7*desired + .15*torch.randn_like(desired)
    wanted = teacher_vectors(reference, desired)
    spectrum = torch.fft.fft(desired, norm='ortho')
    for mark in bands(256, desired.device):
        d, b = (wanted[k][..., mark] for k in ('direct', 'balanced'))
        p, o = split_along(b, spectrum[..., mark])
        torch.testing.assert_close(rms(d), rms(b), atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(rms(p), rms(o), atol=1e-7, rtol=1e-5)
        assert (rms(b) <= CAP + 1e-7).all()
    assert not torch.equal(wanted['direct'], wanted['balanced'])


def test_vector_model_begins_at_identity_and_ignores_source_trace_in_readout():
    torch.manual_seed(83)
    model = ComplexVector()
    measured = torch.randn(2, 6, 256, dtype=torch.complex64)
    quality = torch.rand(2, 6, 1)
    base = measured*.90
    raw = model(measured, base, quality)
    assert torch.count_nonzero(raw) == 0
    truth = teacher_vectors(base, measured)['balanced']
    vector_risk(raw, truth).backward()
    assert all(x.out.weight.grad.norm() > 0 for x in (model.temporal, model.spectral))
    with torch.no_grad():
        model.spectral.out.bias.fill_(.5)
    received = torch.randn(2, 6, 2, 256)
    noise = .05*torch.randn_like(received)

    def baseline(iq):
        return .9*iq, torch.tensor([True, False])

    tracker = PacketRestorer('consensus').eval()
    a, n = readouts(received, noise, baseline, tracker, (model, model))
    b, _ = readouts(received, -noise, baseline, tracker, (model, model))
    assert n == 1 and a.shape == (len(MODES), *received.shape)
    assert torch.equal(a[:5], b[:5])
    assert torch.equal(a[1:, 1], (.9*received[1]).expand_as(a[1:, 1]))
