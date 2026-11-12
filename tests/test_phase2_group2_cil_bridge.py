import time

import torch
import pytest
from torch import nn

from experiments.pamf_phase2_practical.group2_cil_bridge import from_adapted_model, from_protonet, from_radionet, make_anchor, register
from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import _record


@pytest.mark.parametrize(("gradient", "names"), [(float("nan"), "['weight']"), (float("inf"), "['weight']")])
def test_record_identifies_nonfinite_gradient_norm(gradient, names):
    model = nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, gradient)
    trace = []
    with pytest.raises(FloatingPointError, match="training term: gradient_norm") as failure:
        _record(trace, model, {"loss": torch.tensor(1.0)}, None, time.perf_counter(),
                method="mopc_hr", stage="hierarchical", iteration=7)
    assert f"'nonfinite_gradient_tensors': {names}" in str(failure.value)
    assert "'iteration': 7" in str(failure.value)
    if names == "[]":
        assert "'gradient_norm_float64':" in str(failure.value)
        assert "'gradient_absmax_float64':" in str(failure.value)
    else:
        assert "'gradient_norm_float64':" not in str(failure.value)
    assert not trace


class DaddaLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 4)
        self.classifier = nn.Module()
        self.classifier.net = nn.Sequential(nn.ReLU(), nn.Linear(4, 2))

    def forward(self, x):
        z = self.encoder(x)
        return {'local_features': z, 'logits': self.classifier.net(z)}


def test_b_logits_and_state_are_inherited_and_no_query_is_used_for_fit():
    torch.manual_seed(12)
    b = DaddaLike().eval()
    xold = torch.tensor([[0., 0.], [.1, 0.], [1., 1.], [1.1, 1.]])
    yold = torch.tensor([0, 0, 1, 1])
    old = b(xold)['logits'].detach()
    adapted = from_adapted_model(b, 'dadda_sda', old_count=2)
    assert torch.allclose(old, adapted(xold)['tx_logits'])
    anchor = make_anchor(adapted, xold, yold, method='mopc_hr')
    new_x = torch.tensor([[2., 2.], [2.1, 2.], [3., 3.], [3.1, 3.]])
    new_y = torch.tensor([2, 2, 3, 3])
    result = register(anchor, new_x, new_y, epochs=1, seed=123)
    assert result.score(new_x).shape == (4, 4)
    assert anchor.model.fc.out_features == 2
    assert result.protocol_metadata['C_inherits_B_state']
    assert result.protocol_metadata['source_pseudo_targets_preserved'] is False
    assert torch.allclose(old, b(xold)['logits'])


def test_orthogonal_bridge_marks_missing_source_orthogonal_pretraining():
    model = from_adapted_model(DaddaLike().eval(), 'dadda_sda', old_count=2)
    x = torch.tensor([[0., 0.], [1., 1.]])
    anchor = make_anchor(model, x, torch.tensor([0, 1]), method='orthogonal')
    state = register(anchor, torch.tensor([[2., 2.], [3., 3.]]), torch.tensor([2, 3]), epochs=1, seed=12)
    assert state.score(x).shape == (2, 4)
    assert state.protocol_metadata['source_pseudo_targets_preserved'] is False
    assert state.protocol_metadata['old_bias_retained'] is False


def test_rejects_incompatible_backbone_and_class_mismatch():
    with pytest.raises(ValueError):
        from_adapted_model(DaddaLike(), 'not_bound', old_count=2)
    with pytest.raises(ValueError):
        from_adapted_model(DaddaLike(), 'dadda_sda', old_count=6)


def test_dadda_adapter_preserves_actual_active_class_logits():
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import _ActiveClasses
    from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.DADDA.model import DADDANet
    base = DADDANet(num_classes=6, model_variant='conv2d_paper', base_channels=4).eval()
    wrapped = _ActiveClasses(base, list(range(6))).eval()
    samples = torch.randn(2, 2, 256)
    with torch.no_grad():
        expected = wrapped(samples)['logits']
        observed = from_adapted_model(wrapped, 'dadda_sda', old_count=6)(samples)['tx_logits']
    assert torch.allclose(expected, observed, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('profile', ['poster_ft', 'feature_separation_hu2024'])
def test_native_additional_adapters_preserve_b_logits(profile):
    from experiments.pamf_phase2_practical.external_additional import NativeModel
    model = NativeModel(profile, 6, 5).eval()
    samples = torch.randn(2, 2, 256)
    with torch.no_grad():
        expected = model(samples)
        observed = from_adapted_model(model, profile, old_count=6)(samples)['tx_logits']
    assert torch.allclose(expected, observed, rtol=1e-6, atol=1e-7)


def test_protonet_squared_distances_preserved_by_affine_bridge():
    class Proto(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 4)
        def forward(self, x):
            return {'features': self.linear(x)}
    base = Proto().eval()
    old = torch.randn(6, 4)
    x = torch.randn(12, 2)
    bridged = from_protonet(base, old)
    expected = -torch.cdist(base(x)['features'], old).square()
    assert torch.allclose(bridged(x)['tx_logits'], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('profile,method', [('mrior_sda', 'mrior'), ('twostage', 'twostage')])
def test_native_da_adapters_preserve_active_logits(profile, method):
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.adaptation import _ActiveClasses
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.models import factory
    wrapped = _ActiveClasses(factory(method, 6, 5).eval(), list(range(6))).eval()
    samples = torch.randn(2, 2, 256)
    with torch.no_grad():
        expected = wrapped(samples)['tx_logits']
        observed = from_adapted_model(wrapped, profile, old_count=6)(samples)['tx_logits']
    assert torch.allclose(expected, observed, rtol=1e-6, atol=1e-7)


def test_radionet_knn_votes_preserved_in_bridge():
    from experiments.pamf_phase2_practical.external_native import _Scores
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 4)
        def embedding(self, x):
            return self.fc(x)
        def forward(self, x):
            z = self.embedding(x)
            return {'features': z, 'tx_logits': z}
    model = Encoder().eval()
    bank = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    score = _Scores('radionet_ada', decision='knn', support_features=bank,
                    support_labels=labels, knn_k=2, count=3)
    x = torch.randn(12, 2)
    adapted = from_radionet(model, score, old_count=3)
    assert torch.allclose(adapted(x)['tx_logits'], score(model, x), rtol=1e-6, atol=1e-7)


def test_mopc_registration_rng_is_repeatable_and_does_not_modify_caller_rng():
    class StochasticDadda(DaddaLike):
        def __init__(self):
            super().__init__()
            self.classifier.net.insert(1, nn.Dropout(.5))
    adapted = from_adapted_model(StochasticDadda().eval(), 'dadda_sda', old_count=2)
    old = torch.tensor([[0., 0.], [1., 1.]])
    anchor = make_anchor(adapted, old, torch.tensor([0, 1]), method='mopc_hr')
    new = torch.tensor([[2., 2.], [3., 3.]])
    labels = torch.tensor([2, 3])
    torch.manual_seed(345)
    before = torch.random.get_rng_state().clone()
    first = register(anchor, new, labels, seed=45, epochs=1)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(10)
    second = register(anchor, new, labels, seed=45, epochs=1)
    assert torch.equal(first.score(new), second.score(new))
