import torch
from experiments.phase1_dura_orthogonal.local_ssl_iq import (
    PatchIQEncoder, patch_views, patch_cross_view_loss, frozen_patch_feature_loss)


def test_patches_preserve_record_pairing_and_order():
    x = torch.zeros(2, 2, 256)
    x[0, :, :128], x[0, :, 128:] = 1, 2
    x[1, :, :128], x[1, :, 128:] = 3, 4
    patches = patch_views(x)
    assert patches.shape == (4, 2, 128)
    assert patches[:, 0, 0].tolist() == [1, 3, 2, 4]


def test_patch_self_supervision_and_frozen_recovery_gradient():
    torch.manual_seed(7)
    encoder = PatchIQEncoder()
    x, y = torch.randn(4, 2, 256), torch.randn(4, 2, 256)
    loss = patch_cross_view_loss(encoder, x, y)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    encoder.zero_grad(set_to_none=True)
    encoder.eval().requires_grad_(False)
    restored = x.clone().requires_grad_(True)
    clean = y.clone().requires_grad_(True)
    feature = frozen_patch_feature_loss(encoder, restored, clean)
    feature.backward()
    assert restored.grad is not None and torch.isfinite(restored.grad).all()
    assert clean.grad is None and all(p.grad is None for p in encoder.parameters())
