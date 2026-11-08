import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.optimizer import register_optimizer_step_post_hook

from experiments.pamf_phase2_practical import group2_dadda_mopc_diagnostic as diagnostic
from experiments.pamf_phase2_practical import group2_c_backbone_worker
from experiments.pamf_phase2_practical.group2_cil_bridge import AdaptedClassifier, make_anchor
from experiments.pamf_phase2_practical.group2_cross_fit import register_c


@pytest.fixture
def support():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = AdaptedClassifier(nn.Linear(3, 4), nn.Linear(4, 6), None)
        old_x = torch.randn(6, 3)
        anchor = make_anchor(model, old_x, torch.arange(6), method='mopc_hr')
        yield {'mopc_hr': anchor}, torch.randn(10, 3), torch.arange(6, 16)
    torch.set_num_threads(previous_threads)


def test_diagnostic_preserves_training_and_rng(support):
    prepared, x, y = support
    baseline = register_c(prepared, x, y, seed=42, methods=('mopc_hr',))['mopc_hr']
    rng = torch.get_rng_state().clone()
    events = []
    observed = diagnostic.trace_registration(prepared, x, y, 42, events.append)
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in baseline.model.state_dict().items():
        assert torch.equal(observed.model.state_dict()[key], value)
    assert torch.equal(observed.prototypes, baseline.prototypes)
    assert events == [{'event': 'REGISTRATION_RETURNED', 'observed_nonfinite': False, 'completed_updates': 40}]


def test_reports_bad_sgd_update_before_next_forward(support):
    prepared, x, y = support
    events = []

    def corrupt(optimizer, _args, _kwargs):
        with torch.no_grad():
            optimizer.param_groups[0]['params'][0].fill_(float('inf'))

    hook = register_optimizer_step_post_hook(corrupt)
    try:
        with pytest.raises(FloatingPointError):
            diagnostic.trace_registration(prepared, x, y, 42, events.append)
    finally:
        hook.remove()
    assert events[0] == dict(event='FIRST_NONFINITE', boundary='after_sgd', update=1, tensors=['base.weight'])
    assert events[-1]['event'] == 'REGISTRATION_FAILURE'
    assert events[-1]['completed_updates'] == 1


def test_support_rejects_other_capsule_before_reading_samples(tmp_path, monkeypatch):
    def forbidden(*_args):
        pytest.fail('unbound support samples must not be opened')

    monkeypatch.setattr(group2_c_backbone_worker, 'support_only', forbidden)
    splits = tmp_path / 'splits'
    splits.mkdir()
    old_split, new_split = 'a' * 24, 'b' * 24
    (splits / (old_split + '.json')).write_text(json.dumps(dict(capsule_id='bound', k=1, new_count=0)))
    (splits / (new_split + '.json')).write_text(json.dumps(dict(capsule_id='other', k=1, new_count=20)))
    with pytest.raises(ValueError, match='does not match the registered job'):
        diagnostic.support_pair(tmp_path, 'bound', [[1, 20]], old_split, new_split)


def test_support_pair_passes_only_new_class_rows(tmp_path, monkeypatch):
    old_split, new_split = 'a' * 24, 'b' * 24
    splits = tmp_path / 'splits'
    splits.mkdir()
    (splits / (old_split + '.json')).write_text(json.dumps(dict(capsule_id='bound', k=1, new_count=0)))
    (splits / (new_split + '.json')).write_text(json.dumps(dict(capsule_id='bound', k=1, new_count=2)))
    x = np.arange(24).reshape(8, 3)
    labels = np.arange(8)
    ids = np.arange(100, 108)

    def support_only(_capsule, split):
        size = {old_split: 6, new_split: 8}[split]
        return x[:size], labels[:size], ids[:size], {}

    monkeypatch.setattr(group2_c_backbone_worker, 'support_only', support_only)
    old_x, old_y, new_x, new_y, k, n = diagnostic.support_pair(tmp_path, 'bound', [[1, 2]], old_split, new_split)
    assert (k, n) == (1, 2)
    np.testing.assert_array_equal(old_x, x[:6])
    np.testing.assert_array_equal(old_y, labels[:6])
    np.testing.assert_array_equal(new_x, x[6:])
    np.testing.assert_array_equal(new_y, labels[6:])


def test_gradient_diagnostic_distinguishes_float32_overflow_from_nonfinite_values():
    model = nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[1e20]], dtype=torch.float32)
    report = diagnostic.gradient_diagnostic(model)
    assert report['all_elements_finite'] is True
    assert report['legacy_norm_finite'] is False
    assert report['norm_float64'] == pytest.approx(1e20, rel=1e-6)
    model.weight.grad.fill_(float('nan'))
    report = diagnostic.gradient_diagnostic(model)
    assert report['all_elements_finite'] is False
    assert report['norm_float64'] is None
