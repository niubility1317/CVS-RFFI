import torch

from experiments.pamf_lora_main.private_polar import coherent_amplitude, polar_deltas, polar_parts
from experiments.pamf_lora_main.private_transport_bands import named_band_masks


def test_modal_polar_swaps_keep_the_requested_amplitude_and_phase():
    midpoint = torch.tensor([[[3+4j, 1j], [-2j, 2-3j]]])
    full = torch.tensor([[[-2+2j, -1j], [3+0j, 0j]]])
    phase, magnitude = polar_parts(midpoint, full, per_mode=True)
    torch.testing.assert_close(phase.abs(), midpoint.abs())
    torch.testing.assert_close(magnitude.abs(), full.abs())
    torch.testing.assert_close(phase[..., 0].angle(), full[..., 0].angle())
    torch.testing.assert_close(magnitude[..., 0].angle(), midpoint[..., 0].angle())
    torch.testing.assert_close(phase[:, 1, 1], midpoint[:, 1, 1])


def test_vector_phase_keeps_the_frequency_norm_and_donor_direction():
    midpoint = torch.tensor([[[3+4j], [-2j]]])
    full = torch.tensor([[[-2+2j], [3+0j]]])
    phase, magnitude = polar_parts(midpoint, full, per_mode=False)
    norm = lambda x: torch.linalg.vector_norm(x, dim=1, keepdim=True)
    torch.testing.assert_close(norm(phase), norm(midpoint))
    torch.testing.assert_close(phase/norm(phase), full/norm(full))
    torch.testing.assert_close(norm(magnitude), norm(full))
    torch.testing.assert_close(magnitude/norm(magnitude), midpoint/norm(midpoint))


def test_coherent_amplitude_does_not_amplify_reversed_or_unresolved_directions():
    midpoint = torch.tensor([[[2+0j, 2+0j, 2+0j, 2+0j]]])
    donor = torch.tensor([[[1+0j, -1+0j, 0j, 1e-12+0j]]])
    expected = torch.tensor([[[2+0j, -1+0j, 0j, 1e-12+0j]]])
    torch.testing.assert_close(coherent_amplitude(midpoint, donor), expected)


def test_all_polar_edits_keep_public_mean_and_their_band_boundaries():
    generator = torch.Generator().manual_seed(23)
    rank, half, full = [torch.randn(2,6,256,dtype=torch.complex64,generator=generator)
                        for _ in range(3)]
    masks = named_band_masks(rank)
    edits = polar_deltas(rank,half,full,masks)
    assert len(edits) == 6
    for i, edit in enumerate(edits):
        diff = torch.fft.fft(edit-half,norm='ortho')
        torch.testing.assert_close(diff.mean(1), torch.zeros_like(diff[:,0]), atol=1e-6, rtol=0.)
        band = masks[:2].any(0) if i==4 else masks.any(0)
        torch.testing.assert_close(diff[...,~band], torch.zeros_like(diff[...,~band]), atol=1e-6, rtol=0.)
        assert diff[...,band].abs().amax() > .01
