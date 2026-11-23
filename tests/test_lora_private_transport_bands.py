import torch

from experiments.pamf_lora_main.private_transport_bands import band_deltas, named_band_masks


def test_every_band_subset_applies_only_prespecified_frequency_increments():
    gen = torch.Generator().manual_seed(812)
    shape = (2, 6, 256)
    measured = torch.randn(shape, generator=gen, dtype=torch.complex64)
    masks = named_band_masks(measured)
    expected = measured * masks.any(0)
    half = torch.zeros_like(measured)
    full = torch.fft.ifft(expected, norm='ortho')
    rows = band_deltas(half, full, masks)
    assert tuple(rows) == ('000', '100', '010', '001', '110', '101', '011', '111')
    assert rows['000'] is half and rows['111'] is full
    for ident, wave in rows.items():
        use = torch.tensor([v=='1' for v in ident])
        focus = masks[use].any(0)
        torch.testing.assert_close(torch.fft.fft(wave, norm='ortho'),
                                   expected * focus, atol=6e-7, rtol=1e-5)


def test_subsets_preserve_chirp_common_mean_and_do_not_change_carrier():
    gen = torch.Generator().manual_seed(110)
    base = torch.randn((2,6,256),dtype=torch.complex64,generator=gen)
    base = base - base.mean(1,keepdim=True)
    change = torch.randn((2,6,256),dtype=torch.complex64,generator=gen)
    change = change - change.mean(1,keepdim=True)
    masks = named_band_masks(base)
    full = base + torch.fft.ifft(torch.fft.fft(change,norm='ortho') * masks.any(0),norm='ortho')
    for wave in band_deltas(base,full,masks).values():
        torch.testing.assert_close(wave.mean(1),base.mean(1),atol=5e-7,rtol=0.)
        torch.testing.assert_close(wave.mean(-1),base.mean(-1),atol=4e-7,rtol=0.)
