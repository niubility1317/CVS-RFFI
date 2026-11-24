import torch

from experiments.pamf_lora_main.dura_private_projection_source import norm_squared, private, protect_edits


def fixture():
    received = torch.zeros((1, 6, 256), dtype=torch.complex64)
    edit = torch.zeros_like(received)
    mask = torch.zeros(256, dtype=torch.bool)
    mask[3] = True
    axis = torch.tensor([1., -1., 0., 0., 0., 0.])
    received[0,:,3] = axis
    edit[0,:,3] = (-.5+.3j) * axis + torch.tensor([.2, .2, -.4, 0., 0., 0.]) + .2
    edit[0,:,35] = axis
    return received, edit, mask


def test_observed_projection_removes_half_of_negative_radial_only():
    received, edit, mask = fixture()
    received[...,131] = received[...,3] * (0.5**.5)
    (off, all_radial, guarded, equal), stats = protect_edits(received, edit, mask)
    torch.testing.assert_close(stats[0,0], torch.tensor(.5))
    torch.testing.assert_close(guarded, all_radial*.5)
    torch.testing.assert_close(guarded[...,3], .25*received[...,3])
    assert torch.count_nonzero(guarded[...,~mask]) == 0
    torch.testing.assert_close(guarded.mean(1), torch.zeros_like(guarded[:,0]))
    torch.testing.assert_close(norm_squared(private(edit)*mask+guarded), norm_squared(private(edit)*mask+equal))
    torch.testing.assert_close(private(edit+off)*mask, torch.zeros_like(edit))


def test_equal_alias_power_leaves_the_original_edit_and_the_control_unchanged():
    received, edit, mask = fixture()
    received[...,131] = received[...,3]
    (_, _, guarded, equal), stats = protect_edits(received, edit, mask)
    assert torch.count_nonzero(guarded) == 0 and torch.count_nonzero(equal) == 0
    torch.testing.assert_close(stats[0,:2], torch.tensor([0.,1.]))


def test_phase_rotation_positive_gain_and_zero_power_need_no_protection():
    received, edit, mask = fixture()
    edit[...,3] = (.2 + .4j)*received[...,3]
    (_, radial, guarded, equal), _ = protect_edits(received, edit, mask)
    assert all(torch.count_nonzero(value) == 0 for value in (radial,guarded,equal))
    empty = torch.zeros_like(received)
    adjustments, stats = protect_edits(empty, empty, mask)
    assert all(torch.isfinite(value).all() for value in (*adjustments,stats))
