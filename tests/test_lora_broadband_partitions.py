import torch

from experiments.pamf_lora_main.broadband_partitions import deploy_vectors, masks, source_swaps
from experiments.pamf_lora_main.broadband_partitions_source import MODES, readouts
from experiments.pamf_lora_main.broadband_vector import CAP, supported
from experiments.pamf_lora_main.broadband_vector_source import readouts as earlier_readouts
from experiments.pamf_lora_main.complex_budget import chirp_rms
from experiments.pamf_lora_main.dura_repair import PacketRestorer


def test_teacher_swap_keeps_outside_bins_and_matches_delivered_spectral_power():
    torch.manual_seed(56)
    learned = supported(torch.randn(2, 6, 256, dtype=torch.complex64), True)
    truth = supported(torch.randn_like(learned), True)
    truth[0, 0] = 0
    swaps, gap = source_swaps(learned, truth)
    assert gap < 1e-7
    for key, selected in masks(learned).items():
        a, b = swaps[f'known_{key}'], swaps[f'matched_{key}']
        assert torch.equal(a[..., ~selected], learned[..., ~selected])
        assert torch.equal(a[..., selected], truth[..., selected])
        assert torch.equal(b[..., ~selected], learned[..., ~selected])
        torch.testing.assert_close(chirp_rms(b), chirp_rms(learned))


def test_mixing_keeps_the_outer_update_and_joint_cap():
    torch.manual_seed(18)
    narrow = supported(torch.randn(3, 6, 256, dtype=torch.complex64), False)
    broad = supported(torch.randn_like(narrow), True)
    values = deploy_vectors(narrow, broad)
    region = masks(broad)
    for k, selected in (('split32', region['far32']), ('split64', region['65_128'])):
        assert torch.equal(values[k][..., selected], broad[..., selected])
        assert (chirp_rms(values[k])<=CAP+1e-7).all()


def test_final_learned_readouts_match_prior_and_never_depend_on_true_noise():
    torch.manual_seed(42)
    y = torch.randn(2, 6, 2, 256)
    noise = .1*torch.randn_like(y)
    allowed = torch.tensor([True, False])

    def baseline(x):
        return .9*x, allowed

    def model(x, b, q):
        return supported(torch.fft.fft(x-b, norm='ortho'), True)

    tracker = PacketRestorer('consensus').eval()
    a, active, gap = readouts(y, noise, baseline, tracker, (model, model), diagnostic=True)
    b, _ = readouts(y, -noise, baseline, tracker, (model, model))
    prior, _ = earlier_readouts(y, noise, baseline, tracker, (model, model))
    assert active == 1 and gap < 1e-6
    assert torch.equal(a[:10], b[:10])
    assert torch.equal(a[:4], prior[:4])
    assert torch.equal(a[MODES.index('known_all')], prior[7])
    assert torch.equal(a[1:, 1], prior[1, 1].expand_as(a[1:, 1]))
