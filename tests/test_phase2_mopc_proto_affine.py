import torch
import torch.nn.functional as F

from experiments.pamf_phase2_practical.group2_cil_bridge import ProtoClassifier
from experiments.pamf_phase2_practical.group2_proto_mopc_affine import ProtoMoPCAffine, require_equivalent_b


class Features(torch.nn.Module):
    def forward(self, x):
        return {'features': x}


def test_affine_bridge_preserves_proto_probabilities_loss_and_gradients():
    prototypes = torch.tensor([[1., -2., 0.5], [-1., 0.3, 2.], [0.2, 1., -0.5]])
    old = ProtoClassifier(Features(), prototypes)
    new = ProtoMoPCAffine(Features(), prototypes)
    x1 = torch.tensor([[0.3, 1., -0.7], [1.2, -0.2, 0.1]], requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    logits1 = old(x1)['tx_logits']; logits2 = new(x2)['tx_logits']
    assert new.fc.in_features == prototypes.shape[1]
    torch.testing.assert_close(logits2 - logits2.mean(1, keepdim=True),
                               logits1 - logits1.mean(1, keepdim=True))
    torch.testing.assert_close(logits1.softmax(1), logits2.softmax(1))
    y = torch.tensor([0, 2]); loss1 = F.cross_entropy(logits1, y); loss2 = F.cross_entropy(logits2, y)
    torch.testing.assert_close(loss1, loss2)
    loss1.backward(); loss2.backward()
    torch.testing.assert_close(x1.grad, x2.grad)


def test_affine_bridge_avoids_discarded_quadratic_overflow():
    prototypes = torch.tensor([[1., -1.], [-1., 1.]])
    x = torch.tensor([[1.e20, -1.e20]])
    assert not torch.isfinite(ProtoClassifier(Features(), prototypes)(x)['tx_logits']).all()
    assert torch.isfinite(ProtoMoPCAffine(Features(), prototypes)(x)['tx_logits']).all()


def test_equivalence_rejects_near_tie_with_different_argmax():
    import pytest
    a = torch.tensor([[1.e-5, -1.e-5]])
    b = -a
    with pytest.raises(ValueError, match='decision'):
        require_equivalent_b(a, b)
