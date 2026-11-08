import pytest
import torch

from experiments.pamf_phase3_restoration import coupled_feature_response_trial as candidate
from experiments.pamf_phase3_restoration.feature_response_loss import feature_response_loss


class Reader:
    def __init__(self):
        self.grad_modes = []

    def features(self, x):
        self.grad_modes.append(torch.is_grad_enabled())
        return x.flatten(1)[:, :160]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.8))

    def forward(self, x, return_aux=False):
        y = x * self.weight
        return y, {'base_iq': y, 'relative_solve_residual': x.new_zeros(len(x))}


def test_response_loss_uses_aligned_source_only_and_propagates(monkeypatch):
    torch.manual_seed(7)
    x = torch.randn(3, 2, 256)
    clean = torch.randn_like(x)
    delta = torch.full_like(x, .01)
    ids = torch.tensor([1, 2, 3])
    reader, model = Reader(), Model()
    scale = torch.ones(160)
    target = clean.flatten(1)[:, :160].clone()
    monkeypatch.setattr(candidate, 'selected_perturbation', lambda clean, indices, epoch: (delta, indices % 4))
    monkeypatch.setattr(candidate, 'known_parameters', lambda trace: (None, None, None, None))
    monkeypatch.setattr(candidate, 'dense_action', lambda *args: None)
    monkeypatch.setattr(candidate, 'inject', lambda received, delta, action: received + delta)
    def objective(restored, clean):
        return (restored-clean).square().mean((1, 2)), {}
    output, aux, loss, metrics = candidate.response_readout(model, reader, x, clean, target, scale, objective,
        source_indices=ids, epoch=1, source_trace={'received': x})
    base_features = output.flatten(1)[:, :160]
    altered_features = ((x+delta)*model.weight).flatten(1)[:, :160]
    changed_features = (clean+delta).flatten(1)[:, :160]
    expected_feature = feature_response_loss(base_features, altered_features, target, changed_features, scale)
    expected_iq = ((model.weight-1)*delta).square().sum((1,2))/delta.square().sum((1,2))
    torch.testing.assert_close(metrics['source_feature_response_relative_error'], expected_feature)
    torch.testing.assert_close(metrics['source_response_relative_error'], expected_iq, atol=1e-5, rtol=1e-5)
    assert reader.grad_modes == [True, True, False]
    assert torch.isfinite(loss).all() and aux['relative_solve_residual'].shape == (3,)
    loss.mean().backward()
    assert model.weight.grad is not None and model.weight.grad.abs() > 0


def test_trace_must_match_actual_input():
    x = torch.ones(2, 2, 256)
    with pytest.raises(ValueError, match='matched source'):
        candidate.response_readout(None, None, x, x, None, None, None, source_indices=None,
                                   epoch=1, source_trace={'received': x+1})


def test_fixed_recipe_keeps_scratch_and_separate_control():
    cfg = candidate.configuration(candidate.ARM)
    assert cfg['epochs'] == 30 and cfg['updates'] == 7410
    assert cfg['training_stages'] == 1 and cfg['ancestor_checkpoints'] == []
    assert cfg['F0_forward_rows_per_source_record'] == 3
    assert cfg['source_response_weights'] == {'IQ': .01, 'F0': .01}
    assert cfg['TX_labels_read'] == 0 and cfg['target_training_reads'] == 0
