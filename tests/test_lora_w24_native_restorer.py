import torch

from experiments.pamf_lora_main.w24_native_restorer import NativeRestorer, source_loss


def test_native_loss_trains_both_stages_and_preserves_bypass():
    rng = torch.Generator().manual_seed(4)
    clean = torch.randn(1, 8, 1024, dtype=torch.complex64, generator=rng)
    model = NativeRestorer()
    identity_input = clean.clone()
    assert torch.equal(model.restore(identity_input, threshold=float('inf')), clean)
    loss = source_loss(model, clean, rng)
    loss.backward()
    assert torch.isfinite(loss)
    for stage in (model.coarse, model.prior):
        gradients = [p.grad for p in stage.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert any(grad.abs().sum() > 0 for grad in gradients)
