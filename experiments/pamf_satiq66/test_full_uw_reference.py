import torch
from torch.nn import functional as F

from experiments.pamf_satiq66.model import UWResponse, full_downlink_reference


def test_full_downlink_reference_keeps_the_transition_to_the_twelfth_symbol():
    iq = full_downlink_reference()
    assert iq.shape == (1, 2, 11000)
    expected = torch.tensor([1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1])
    assert torch.equal(iq[0, 0, ::1000].sign(), expected)
    assert iq[0, 0, 10400] > 0
    assert iq[0, 0, 10600] < 0


def test_full_reference_uses_the_tail_without_changing_the_response_width():
    grid = F.interpolate(full_downlink_reference(), size=256, mode="linear", align_corners=False)
    aligned_prefix = UWResponse(1, waveform_samples=11000)
    full = UWResponse(1, waveform_samples=11000, reference_grid=grid[0])
    different_tail = grid.clone()
    different_tail[..., len(aligned_prefix.reference):] *= -1

    assert full(grid).shape == aligned_prefix(grid).shape == (1, 29)
    torch.testing.assert_close(aligned_prefix(grid), aligned_prefix(different_tail))
    assert torch.isfinite(full(different_tail)).all()
    assert not torch.allclose(full(grid), full(different_tail))
