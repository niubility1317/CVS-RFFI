import torch

from model_dual_cvsincnet import SatAnchorIdentityAdapter


def test_zero_initialized_adapter_preserves_features_and_logits():
    adapter = SatAnchorIdentityAdapter(feature_dim=6, num_classes=3, rank=2)
    feature = torch.randn(4, 6)

    adapted, correction = adapter(feature, detach_backbone=False)

    assert torch.equal(adapted, feature)
    assert torch.count_nonzero(correction).item() == 0


def test_unlabeled_detach_blocks_backbone_but_updates_adapter_tail():
    adapter = SatAnchorIdentityAdapter(feature_dim=6, num_classes=3, rank=2)
    feature = torch.randn(4, 6, requires_grad=True)

    adapted, correction = adapter(feature, detach_backbone=True)
    loss = adapted.square().mean() + correction.sum()
    loss.backward()

    assert feature.grad is None
    assert adapter.up.weight.grad is not None
    assert adapter.logit_correction.weight.grad is not None
    assert adapter.logit_correction.weight.grad.abs().sum().item() > 0


def test_labeled_path_keeps_full_backbone_gradient():
    adapter = SatAnchorIdentityAdapter(feature_dim=6, num_classes=3, rank=2)
    feature = torch.randn(4, 6, requires_grad=True)

    adapted, correction = adapter(feature, detach_backbone=False)
    (adapted.square().mean() + correction.sum()).backward()

    assert feature.grad is not None
    assert feature.grad.abs().sum().item() > 0
