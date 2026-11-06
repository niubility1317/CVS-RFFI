import torch
from experiments.phase1_dura_orthogonal.joint_ssl_iq import SourceIQEncoder, cross_view_ssl_loss, frozen_feature_loss


def test_encoder_uses_no_batch_statistics_at_inference():
    torch.manual_seed(7)
    encoder = SourceIQEncoder()
    x = torch.randn(4, 2, 256)
    batch = encoder(x)
    single = torch.cat([encoder(x[i:i+1]) for i in range(4)])
    torch.testing.assert_close(batch, single, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batch.norm(dim=1), torch.ones(4), rtol=1e-5, atol=1e-6)


def test_cross_view_ssl_loss_has_finite_gradients_without_labels():
    torch.manual_seed(8)
    a = torch.randn(32, 16, requires_grad=True)
    b = torch.randn(32, 16, requires_grad=True)
    loss = cross_view_ssl_loss(a, b)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.count_nonzero(a.grad)
    assert torch.isfinite(b.grad).all() and torch.count_nonzero(b.grad)


def test_frozen_feature_loss_backpropagates_only_to_restored_iq():
    torch.manual_seed(9)
    encoder = SourceIQEncoder().requires_grad_(False).eval()
    restored = torch.randn(3, 2, 256, requires_grad=True)
    clean = torch.randn(3, 2, 256, requires_grad=True)
    loss = frozen_feature_loss(encoder, restored, clean)
    loss.backward()
    assert torch.isfinite(restored.grad).all() and torch.count_nonzero(restored.grad)
    assert clean.grad is None and all(p.grad is None for p in encoder.parameters())
