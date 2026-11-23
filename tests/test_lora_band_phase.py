import torch

from experiments.pamf_lora_main.band_phase import CalibratedPacketNear, phase_risk
from experiments.pamf_lora_main.broadband_vector import supported
from experiments.pamf_lora_main.denoise_components import bands
from experiments.pamf_lora_main.near_spectrum import near_error


def test_matching_frame_and_band_specific_transport():
    torch.manual_seed(17)
    y = torch.randn(2, 6, 256, dtype=torch.complex64)
    q = .7*torch.ones(2, 6, 1)
    frozen = supported(torch.randn_like(y), True)
    clean = torch.roll(y, 1, dims=-1) * .9
    model = CalibratedPacketNear()
    first, estimated = model(y, .8*y, q, frozen, phase_readout=True)
    assert torch.equal(first, frozen)
    assert len(estimated) == 2
    assert not torch.equal(estimated[0][0], estimated[1][0])
    for unit, reliability in estimated:
        torch.testing.assert_close(unit, unit.conj().transpose(-1, -2))
        torch.testing.assert_close(reliability, reliability.transpose(-1, -2))
        torch.testing.assert_close(unit.diagonal(dim1=-2, dim2=-1).real, torch.ones(2, 6))
    (near_error(first, .8*y, clean)+.08*phase_risk(estimated, clean).mean()).backward()
    assert model.context.out.weight.grad.norm() > 0
    assert all(head[-1].weight.grad.norm() > 0 for head in model.calibrator.estimators)

    with torch.no_grad():
        model.context.out.weight.fill_(.1)
    result = model(y, .8*y, q, frozen)
    mask = torch.zeros(256, dtype=torch.bool)
    for band in bands(256, y.device):
        mask |= band
        torch.testing.assert_close(result[..., band].norm(dim=-1), frozen[..., band].norm(dim=-1))
    assert torch.equal(result[..., ~mask], frozen[..., ~mask])
    assert not torch.equal(result, frozen)
