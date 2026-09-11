import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvsrffi.anchored_geometry import AnchoredMetricHead, OrdinaryAngleHead


def sample():
    torch.manual_seed(81)
    return torch.randn(12, 8), torch.randn(3, 8)


def test_identity_scale_learning_and_frozen_directions():
    h, w = sample(); head = AnchoredMetricHead(w, 30., rank=4)
    ref = 30 * F.normalize(h, dim=-1, eps=1e-4) @ F.normalize(w, dim=-1, eps=1e-4).T
    torch.testing.assert_close(head(h), ref, atol=1e-5, rtol=1e-5)
    F.cross_entropy(head(h), torch.arange(len(h)) % 3).backward()
    assert head.b.grad.abs().sum() > 0
    assert head.w0.grad is None
    assert sum(p.numel() for p in head.parameters()) == 8 * 4 + 4


def test_lowrank_dense_values_gradients_scale_and_condition():
    h, w = sample(); head = AnchoredMetricHead(w, 30., rank=4).double()
    with torch.no_grad(): head.b.copy_(torch.tensor([.5, -.7, 1., -.3]))
    h = h.double().requires_grad_(); q, a = head.spectral()
    m = torch.eye(8, dtype=torch.double) + (q * torch.expm1(a)) @ q.T
    ref = 30 * (h @ m @ head.w0.T) / ((h @ m * h).sum(-1).sqrt()[:, None] * (head.w0 @ m * head.w0).sum(-1).sqrt()[None])
    torch.testing.assert_close(head(h), ref)
    g = torch.autograd.grad(head(h).square().mean(), (h, head.b, head.B), retain_graph=True)
    rg = torch.autograd.grad(ref.square().mean(), (h, head.b, head.B))
    for left, right in zip(g, rg): torch.testing.assert_close(left, right)
    torch.testing.assert_close(head(h * 4), head(h))
    assert head.geometry_diagnostics()['condition_number'] <= 2 + 1e-6
    assert torch.linalg.eigvalsh(m).min() > 0


def test_export_permutation_and_numeric_failures():
    h, w = sample(); head = AnchoredMetricHead(w, 30., rank=4)
    with torch.no_grad(): head.b.add_(.2)
    restored = AnchoredMetricHead.from_export(head.export_state())
    torch.testing.assert_close(restored(h), head(h))
    permutation = torch.tensor([2, 0, 1])
    other = AnchoredMetricHead(w[permutation], 30., rank=4)
    with torch.no_grad(): other.B.copy_(head.B); other.b.copy_(head.b)
    torch.testing.assert_close(other(h), head(h)[:, permutation])
    assert torch.isfinite(head(torch.zeros_like(h))).all()
    with pytest.raises(ValueError, match='finite'): head(h * float('nan'))
    with torch.no_grad(): head.B.zero_()
    with pytest.raises(ValueError, match='rank'): head(h)


def test_ordinary_angle_control_has_trainable_weights_only():
    h, w = sample(); head = OrdinaryAngleHead(w, 30.)
    assert set(dict(head.named_parameters())) == {'weight'}
    torch.testing.assert_close(head(h), 30 * F.normalize(h, dim=-1) @ F.normalize(w, dim=-1).T)
