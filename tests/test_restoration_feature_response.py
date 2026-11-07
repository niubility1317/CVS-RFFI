import pytest
import torch

from experiments.pamf_phase3_restoration.feature_response_loss import feature_response_loss


def test_exact_feature_direction_has_zero_loss_and_scale_invariance():
    source = torch.arange(320, dtype=torch.float32).reshape(2, 160) / 1000 + 0.1
    change = torch.linspace(-0.05, 0.05, 160).expand_as(source)
    base = source + 0.2
    scale = torch.linspace(0.5, 2.0, 160)
    loss = feature_response_loss(base, base + change, source, source + change, scale)
    torch.testing.assert_close(loss, torch.zeros(2), atol=1e-12, rtol=0)
    shifted = feature_response_loss(base, base + 2 * change, source, source + change, scale)
    shifted_scaled = feature_response_loss(3 * base, 3 * (base + 2 * change),
                                           3 * source, 3 * (source + change), 9 * scale)
    torch.testing.assert_close(shifted, shifted_scaled, rtol=1e-5, atol=1e-6)


def test_guidance_differentiates_both_outputs_and_freezes_source_reference():
    source = torch.ones(2, 160, requires_grad=True)
    changed_source = torch.full((2, 160), 1.02, requires_grad=True)
    base = torch.full((2, 160), 0.7, requires_grad=True)
    changed = torch.full((2, 160), 0.71, requires_grad=True)
    scale = torch.ones(160, requires_grad=True)
    loss = feature_response_loss(base, changed, source, changed_source, scale)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and (loss > 0).all()
    assert base.grad is not None and changed.grad is not None
    assert torch.isfinite(base.grad).all() and base.grad.abs().sum() > 0
    assert torch.isfinite(changed.grad).all() and changed.grad.abs().sum() > 0
    assert source.grad is None and changed_source.grad is None and scale.grad is None


@pytest.mark.parametrize('bad_scale', [torch.zeros(160), torch.full((160,), float('nan'))])
def test_rejects_nonpositive_or_nonfinite_source_scale(bad_scale):
    value = torch.ones(2, 160)
    with pytest.raises(ValueError):
        feature_response_loss(value, value, value, value, bad_scale)


def test_zero_reference_response_remains_finite():
    source = torch.zeros(2, 160)
    base = torch.zeros(2, 160)
    changed = torch.full((2, 160), 0.01)
    loss = feature_response_loss(base, changed, source, source, torch.ones(160))
    assert torch.isfinite(loss).all() and (loss > 0).all()


@pytest.mark.parametrize('missing_trace', [True, False])
def test_response_training_rejects_missing_or_mismatched_source_trace(missing_trace):
    from experiments.pamf_phase3_restoration import feature_response_trial as subject

    received = torch.zeros(2, 2, 256)
    trace = None if missing_trace else {'received': torch.ones_like(received)}
    with pytest.raises(ValueError, match='source simulator trace'):
        subject.response_readout(None, None, received, None, None, None, None,
                                 source_indices=torch.arange(2), epoch=1,
                                 priors=None, assignment=None, source_trace=trace)


def test_training_path_uses_paired_feature_differences_without_classifier(monkeypatch):
    from experiments.pamf_phase3_restoration import feature_response_trial as subject

    class Reader:
        def features(self, x):
            return x.flatten(1)[:, :160]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detail = torch.nn.Module()
            self.detail.output = torch.nn.Linear(1, 1, bias=False)
            self.detail.mean_output = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.detail.output.weight.fill_(0.1)
                self.detail.mean_output.weight.fill_(0.2)

        def source_readout(self, received, *, prior=None):
            a = self.detail.output.weight.reshape(())
            b = self.detail.mean_output.weight.reshape(())
            return received + a * received.square() + b * received.pow(3), {
                'relative_solve_residual': torch.zeros(len(received))}

        def _readout(self, received, prior):
            return self.source_readout(received, prior=prior)

    def joint(model, reader, received, clean, features, scale, objective, **kwargs):
        restored, aux = model.source_readout(received)
        return restored, aux, torch.zeros(len(received)), {}

    monkeypatch.setattr(subject.original.control, 'joint_readout', joint)
    monkeypatch.setattr(subject.original, 'selected_perturbation',
                        lambda clean, indices, epoch: (torch.full_like(clean, 0.01), torch.zeros(len(clean), dtype=torch.long)))
    monkeypatch.setattr(subject.original, 'known_parameters', lambda trace: (None, None, None, None))
    monkeypatch.setattr(subject.original, 'dense_action', lambda *args: None)
    monkeypatch.setattr(subject.original, 'inject', lambda received, delta, action: received + delta)
    monkeypatch.setattr(subject, 'choose_prior', lambda *args: None)

    reader, model = Reader(), Model()
    clean = torch.full((2, 2, 256), 0.8)
    received = torch.full_like(clean, 0.6)
    target = reader.features(clean)
    _, _, loss, terms = subject.response_readout(
        model, reader, received, clean, target, torch.ones(160), None,
        source_indices=torch.arange(2), epoch=0, priors=None, assignment=None,
        source_trace={'received': received})
    assert torch.isfinite(loss).all()
    assert (terms['source_feature_response_relative_error'] > 0).all()
    assert model.response_smoke_report['actual_source_feature_response_gradients']
    loss.sum().backward()
    assert model.detail.output.weight.grad.abs().sum() > 0
    assert model.detail.mean_output.weight.grad.abs().sum() > 0
