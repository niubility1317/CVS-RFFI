"""The pre-unit readout must share the existing constant-condition forward."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from experiments.pamf_phase3_open_world import star_physical_open_model as api
from experiments.pamf_phase3_open_world import star_reference_open_model as reference
from test_pamf_phase3_star_reference_open_model import fixture, hooks


def test_preunit_capture_is_same_forward_exact_original_outputs_and_no_parameters():
    model, x = fixture(); reference_model = deepcopy(model)
    api.install(model); reference.install(reference_model, use_reference=False)
    reader = model.id_backbone.encoder
    before = (reader.core.id_backbone.calls, reader.response.calls, reader.sequence_pool.calls)
    original = reference.forward(reference_model, x)
    result = api.forward(model, x)
    assert set(result) == set(original) | {'open_raw64'}
    for key in original: torch.testing.assert_close(result[key], original[key], atol=0, rtol=0)
    assert result['open_raw64'].shape == (len(x), 64) and result['open_raw64'].dtype == torch.float32
    torch.testing.assert_close(F.normalize(result['open_raw64'], dim=1), result['open64'], atol=0, rtol=0)
    assert (reader.core.id_backbone.calls, reader.response.calls, reader.sequence_pool.calls) == tuple(v+1 for v in before)
    assert model.open_world.use_reference is False
    assert api.parameter_report(model) == reference.parameter_report(reference_model)
    assert api.parameter_report(model)['active_parameters'] == 315106
    assert api.parameter_report(model)['training_container_parameters'] == 982835
    assert set(model.state_dict()) == set(reference_model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference_model.state_dict()[name], atol=0, rtol=0)


def test_raw_feature_gradients_reach_native_temporal_branch_and_projector():
    model, x = fixture(); api.install(model)
    initial = hooks(model)
    result = api.forward(model, x)
    weights = torch.linspace(-1, 1, 64)
    (result['open_raw64'] * weights).sum().backward()
    for p in (model.id_backbone.encoder.core.id_backbone.t2[0].weight,
              model.open_world.projector[0].weight, model.open_world.projector[-1].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert hooks(model) == initial
    api.clear(model); assert model._open_world_last is None


def test_temporary_hook_cleans_on_original_or_post_capture_failure_and_no_grad(monkeypatch):
    model, x = fixture(); api.install(model); initial = hooks(model)
    model.id_backbone.encoder.fail = True
    with pytest.raises(RuntimeError, match='synthetic feature failure'): api.forward(model, x)
    assert hooks(model) == initial and model._open_world_last is None
    model.id_backbone.encoder.fail = False
    original = model.id_backbone.encoder.classify_features
    def fail(z): raise RuntimeError('head failure after raw capture')
    monkeypatch.setattr(model.id_backbone.encoder, 'classify_features', fail)
    with pytest.raises(RuntimeError, match='head failure'): api.forward(model, x)
    assert hooks(model) == initial and model._open_world_last is None
    monkeypatch.setattr(model.id_backbone.encoder, 'classify_features', original)
    with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        result = api.forward(model, x)
    assert result['open_raw64'].dtype == torch.float32 and not result['open_raw64'].requires_grad
    assert hooks(model) == initial


def test_raw_capture_is_per_record_and_replaced_between_calls():
    model, x = fixture(); api.install(model); model.train()
    first = api.forward(model, x)
    rows = [api.forward(model, value[None])['open_raw64'] for value in x]
    torch.testing.assert_close(torch.cat(rows), first['open_raw64'], atol=2e-6, rtol=1e-5)
    second = api.forward(model, x + .1)
    assert first['open_raw64'] is not second['open_raw64']
    assert second['open_raw64'].requires_grad
