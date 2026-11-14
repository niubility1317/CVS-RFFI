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


def test_opt_in_affine_mopc_keeps_orthogonal_bridge_unchanged():
    from experiments.pamf_phase2_practical.group2_cross_fit import prepare_b
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.models import ProtoNet
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = ProtoNet(6).eval()
    x = torch.randn(6, 2, 256)
    y = torch.arange(6)
    prototypes = torch.randn(6, 256) * .001
    feature = lambda current, inputs: current(inputs)['features']
    original = prepare_b(model, 'protonet_cda', feature, x, y, old_count=6, old_prototypes=prototypes)
    fixed = prepare_b(model, 'protonet_cda', feature, x, y, old_count=6, old_prototypes=prototypes,
                      mopc_bridge='proto_affine_v1')
    assert original['mopc_hr'].model.fc.in_features == 257
    assert fixed['mopc_hr'].model.fc.in_features == 256
    assert fixed['mopc_hr'].protocol_metadata['mopc_bridge'] == 'proto_affine_v1'
    assert fixed['orthogonal'].model.fc.in_features == 257
    for name, value in original['orthogonal'].model.state_dict().items():
        torch.testing.assert_close(value, fixed['orthogonal'].model.state_dict()[name], rtol=0, atol=0)


def test_affine_prediction_rows_are_distinct_from_legacy_results():
    import pytest
    from experiments.pamf_phase2_practical.group2_worker import require_all_rows
    from experiments.pamf_phase2_practical.clean_score import expected_rows
    job = dict(points=[[5, 2]], profile='protonet_cda', methods=['mopc_hr'],
               mopc_bridge='proto_affine_v1')
    row = dict(k=5, new_count=2, arm='protonet_cda__mopc_hr_affine_v1')
    require_all_rows([row], job['points'], job['profile'], job['methods'], mopc_bridge=job['mopc_bridge'])
    assert ('protonet_cda__mopc_hr_affine_v1', 'C', 5, 2, 'P0') in expected_rows('hybrid', job)
    with pytest.raises(ValueError, match='Incomplete'):
        require_all_rows([row], job['points'], job['profile'], job['methods'])


def test_saved_b_rejects_a_different_mopc_bridge_version(tmp_path):
    import pytest
    from types import SimpleNamespace
    from experiments.pamf_phase2_practical.group2_b_state import _binding, load_b_state
    base = dict(key='k', profile='protonet_cda', seed=7, capsule='capsules/P/M/R')
    anchor = SimpleNamespace(method='mopc_hr', model=torch.nn.Linear(2, 2),
                             prototypes=torch.zeros(2, 2), protocol_metadata={'mopc_bridge': 'proto_affine_v1'})
    path = tmp_path/'b.pt'
    torch.save(dict(version=1, binding=_binding(base, 5, 'old', 11), anchor=anchor), path)
    with pytest.raises(ValueError, match='bridge'):
        load_b_state(path, base, 5, 'old', 11, 'cpu')
    assert load_b_state(path, dict(base, mopc_bridge='proto_affine_v1'), 5, 'old', 11, 'cpu')['anchor']
