import json
import numpy as np
import pytest
import time
import torch
from torch import nn

from experiments.pamf_phase2_practical import group1_source as source
from experiments.pamf_phase2_practical import group1_target as target


@pytest.fixture(autouse=True)
def use_one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_orthogonal_pseudo_confidence_can_accept_and_backpropagate():
    raw = torch.tensor([[1., -1., -1., -1., -1., -1.]], requires_grad=True)
    scaled = source.pseudo_label_scores('orthogonal', raw)
    confidence, label = scaled.softmax(1).max(1)
    assert confidence.item() > .95
    assert label.item() == 0
    torch.nn.functional.cross_entropy(scaled, label).backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum().item() > 0
    torch.testing.assert_close(source.pseudo_label_scores('mopc_hr', raw), raw)


def test_orthogonal_actual_training_uses_tempered_pseudo_labels(tmp_path, monkeypatch):
    from experiments.practical_full_2x2 import cache
    class Data:
        def __init__(self, hidden=False):
            labels = torch.arange(6).repeat_interleave(2)
            self.x = torch.zeros(12, 2, 256)
            self.x[torch.arange(12), 0, labels] = 1.
            self.metadata = {'rx_i': np.ones(12, dtype=int)}
            if not hidden:
                self.metadata['y'] = labels.numpy()
        def __len__(self):
            return len(self.x)
    datasets = {'L_s': Data(), 'U_s': Data(True), 'V': Data()}
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(6))
            self.register_buffer('pseudo_targets', torch.eye(6))
        def forward(self, x):
            return torch.nn.functional.normalize(x[:, 0, :6] @ self.weight, dim=1)
    class Reader:
        wait_seconds = 0.
        def __init__(self, config):
            pass
    monkeypatch.setattr(cache, 'Reader', Reader)
    monkeypatch.setattr(source, 'source_data', lambda config, role: datasets[role])
    monkeypatch.setattr(source, 'row', lambda profile: dict(method='orthogonal', epochs=2, PL_start_epoch=2))
    monkeypatch.setattr(source, 'model_for', lambda method, device: Model())
    monkeypatch.setattr(source, 'logits', lambda model, method, x: model(x))
    monkeypatch.setattr(source, 'supervised', lambda model, method, x, y, rx: torch.nn.functional.cross_entropy(model(x), y))
    monkeypatch.setattr(source, 'augment', lambda reader, ids, epoch, epochs, device:
                        (datasets['L_s'].x[ids].to(device), {'applied_count': len(ids)}))
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'source_rxs': [1]}), encoding='utf-8')
    source._train_native(dict(source_contract=str(contract), source_cache='synthetic'),
                         'orthogonal', 42, tmp_path / 'output', 'cpu')
    epochs = [json.loads(line) for line in (tmp_path / 'output/epochs.jsonl').read_text().splitlines()]
    assert epochs[0]['PL_accepted'] == 0
    assert epochs[1]['PL_accepted'] > 0 and epochs[1]['PL_loss'] > 0


def _lora_bundle():
    features = torch.arange(6 * 8 * 2, dtype=torch.float32).reshape(48, 2)
    labels = torch.arange(6).repeat_interleave(8)
    return dict(profile='lora_triplet_knn', method='lora_triplet_knn', model=nn.Linear(2, 2),
                classes=tuple(range(6)), resources={},
                native_state=dict(source_enrollment_features=features, source_enrollment_y=labels))


def test_lora_gallery_is_balanced_seeded_and_preserves_source(monkeypatch):
    monkeypatch.setattr(target, 'embedding', lambda model, method, x: x)
    bundle = _lora_bundle()
    before = bundle['native_state']['source_enrollment_features'].clone()
    x = torch.tensor([[100., 1.], [101., 1.], [200., 1.], [201., 1.]])
    y = torch.tensor([6, 6, 7, 7])
    a = target.fit_C({}, bundle, x, y, (6, 7), 123, lambda _: None)
    b = target.fit_C({}, bundle, x, y, (6, 7), 123, lambda _: None)
    native = a['native_state']
    assert torch.bincount(native['enrollment_y']).tolist() == [2] * 8
    torch.testing.assert_close(native['enrollment_features'], b['native_state']['enrollment_features'])
    torch.testing.assert_close(native['enrollment_features'][-4:], x)
    torch.testing.assert_close(bundle['native_state']['source_enrollment_features'], before)
    assert a['resources']['source_gallery_count'] == 12
    assert a['resources']['source_gallery_available'] == 48
    assert a['resources']['knn_k'] == 3


def test_lora_rejects_unbalanced_new_support(monkeypatch):
    monkeypatch.setattr(target, 'embedding', lambda model, method, x: x)
    with pytest.raises(ValueError, match='equal'):
        target.fit_C({}, _lora_bundle(), torch.ones(3, 2), torch.tensor([6, 6, 7]),
                     (6, 7), 123, lambda _: None)


def test_lora_one_shot_vote_can_select_the_nearest_new_class(monkeypatch):
    monkeypatch.setattr(target, 'embedding', lambda model, method, x: x)
    monkeypatch.setattr(target.extra, 'native_features', lambda model, x: x)
    bundle = _lora_bundle()
    bundle['native_state']['source_enrollment_features'] += 100.
    state = target.fit_C({}, bundle, torch.tensor([[0., 0.], [50., 50.]]),
                         torch.tensor([6, 7]), (6, 7), 123, lambda _: None)
    assert state['resources']['knn_k'] == 1
    assert state['score_fn'](state['model'], torch.tensor([[.01, .01]])).argmax(1).item() == 6


@pytest.mark.parametrize('k', [1, 5, 10, 20])
@pytest.mark.parametrize('new_count', [1, 5, 10, 20])
def test_lora_balances_each_configured_k_and_n(monkeypatch, k, new_count):
    monkeypatch.setattr(target, 'embedding', lambda model, method, x: x)
    bundle = _lora_bundle()
    bundle['native_state']['source_enrollment_features'] = torch.randn(6 * 25, 2)
    bundle['native_state']['source_enrollment_y'] = torch.arange(6).repeat_interleave(25)
    labels = torch.arange(6, 6 + new_count).repeat_interleave(k)
    state = target.fit_C({}, bundle, torch.randn(len(labels), 2), labels,
                         tuple(range(6, 6 + new_count)), 123, lambda _: None)
    assert torch.bincount(state['native_state']['enrollment_y']).tolist() == [k] * (6 + new_count)
    assert state['resources']['knn_k'] == min(15, 2 * k - 1)


def test_mopc_group2_explicitly_selects_public_repository(monkeypatch):
    from experiments.pamf_phase2_practical import group2_cil_bridge as bridge
    class Anchor:
        classes = (0, 1)
        method = 'mopc_hr'
        protocol_metadata = {'old_support_k': 1}
    observed = {}
    def fake(anchor, new_x, new_y, ids, *, config):
        observed['semantics'] = config.mopc_semantics
        return anchor
    monkeypatch.setattr(bridge, 'register_new_classes', fake)
    bridge.register(Anchor(), torch.ones(1, 2), torch.tensor([2]), seed=7, epochs=1)
    assert observed['semantics'] == 'public_code'


def test_mopc_group1_recipe_identifies_public_repository():
    assert source.recipe('mopc_hr')['C_semantics'] == 'public_code'


def test_large_finite_gradients_do_not_overflow_the_norm_observation():
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration
    model = nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, 1.e30)
    trace = []
    registration._record(trace, model, {'loss': torch.tensor(1.)}, None, time.perf_counter(),
                         phase='C', method='mopc_hr')
    assert trace[0]['gradient_norm'] > 1.e30
    model.weight.grad[0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='nonfinite'):
        registration._record([], model, {'loss': torch.tensor(1.)}, None, time.perf_counter(),
                             phase='C', method='mopc_hr')


@pytest.mark.parametrize('seed', [1729, 1731, 1733])
def test_public_mopc_full_budget_on_synthetic_protonet(seed):
    from experiments.pamf_phase2_practical.group2_cil_bridge import from_protonet, make_anchor, register
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite.models import ProtoNet
    torch.manual_seed(seed)
    model = from_protonet(ProtoNet(6).eval(), torch.randn(6, 256) * .001)
    old_x = torch.randn(60, 2, 256)
    old_y = torch.arange(6).repeat_interleave(10)
    anchor = make_anchor(model, old_x, old_y, method='mopc_hr')
    new_x = torch.randn(100, 2, 256)
    new_y = torch.arange(6, 16).repeat_interleave(10)
    result = register(anchor, new_x, new_y, seed=42)
    assert len(result.loss_trace) == 160
    assert result.protocol_metadata['mopc_version'] == 'public_code_semantics_matched_IQ_extension'
    assert all(torch.isfinite(value).all() for value in result.model.state_dict().values())
