import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.noise_split_diagnostic import components, exact_readouts


def test_frequency_partition_and_common_private_add_to_noncarrier():
    rng = torch.Generator().manual_seed(20261124)
    exact = torch.randn(2,6,256,dtype=torch.complex64,generator=rng)
    common,private,noncarrier,*frequency = components(exact)
    torch.testing.assert_close(common+private,noncarrier)
    torch.testing.assert_close(sum(frequency),noncarrier)
    assert not noncarrier[...,0].any()
    torch.testing.assert_close(private.mean(1),torch.zeros_like(private[:,0]),atol=3e-7,rtol=0)


def test_counterparts_are_matched_waveforms_and_clean_is_exact():
    rng = torch.Generator().manual_seed(121)
    tracker = PacketRestorer('consensus').eval()
    clean = tracker.reference.chirp * (1. + .15*torch.randn(2,6,256,dtype=torch.complex64,generator=rng))
    eps = .3*torch.randn(clean.shape,dtype=torch.complex64,generator=rng)
    eps[0].zero_()
    raw = real_iq(clean+eps)
    out,active,details = exact_readouts(raw,real_iq(eps),tracker)
    assert active == 1 and out.shape == (10,*raw.shape) and details.shape == (2,9)
    assert torch.equal(out[:,0],raw[0].expand_as(out[:,0]))
    torch.testing.assert_close(complex_iq(out[3]),clean)
    y = complex_iq(raw)
    torch.testing.assert_close(complex_iq(out[-2]).abs(),y.abs())
    torch.testing.assert_close(complex_iq(out[-1]).abs(),complex_iq(out[2]).abs())
