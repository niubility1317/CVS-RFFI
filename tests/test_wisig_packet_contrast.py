import torch

from experiments.pamf_lora_main.dura_real_gain_wisig_packet_contrast import pair_contrastive


def test_packet_contrast_prefers_paired_iq_and_backpropagates():
    torch.manual_seed(27)
    clean = torch.randn(8, 256, dtype=torch.complex64)
    restored = (clean + .2 * torch.randn_like(clean)).requires_grad_()
    aligned = pair_contrastive(restored, clean)
    mismatched = pair_contrastive(restored, clean.roll(1, dims=0))
    assert aligned < mismatched
    aligned.backward()
    assert restored.grad is not None
    assert torch.isfinite(restored.grad).all()
    assert restored.grad.abs().sum() > 0
