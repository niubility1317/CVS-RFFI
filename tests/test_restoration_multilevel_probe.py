import pytest
import torch
from torch import nn


class TinyReader(nn.Module):
    def __init__(self, repeat=False, fail=False):
        super().__init__()
        self.core = nn.Module()
        self.core.id_backbone = nn.Module()
        self.core.id_backbone.time_down = nn.AvgPool1d(2)
        self.core.id_backbone.t2 = nn.Module()
        self.core.id_backbone.t2.pool = nn.AvgPool1d(2)
        self.repeat = repeat
        self.fail = fail

    def features(self, x):
        y = self.core.id_backbone.time_down(x)
        if self.repeat:
            self.core.id_backbone.time_down(x)
        if self.fail:
            raise RuntimeError('source failure')
        z = self.core.id_backbone.t2.pool(y)
        return z.mean(dim=-1)


def test_capture_preserves_input_gradients_and_removes_hooks():
    from experiments.pamf_phase3_restoration.source_multilevel_gradient_probe import captured_features
    reader = TinyReader()
    x = torch.randn(1, 2, 16, requires_grad=True)
    output = captured_features(reader, x)
    assert set(output) == {'shallow', 'middle', 'final'}
    for value in output.values():
        gradient = torch.autograd.grad(value.square().mean(), x, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert not reader.core.id_backbone.time_down._forward_hooks
    assert not reader.core.id_backbone.t2.pool._forward_hooks


def test_capture_rejects_ambiguous_layer_execution_and_removes_hooks():
    from experiments.pamf_phase3_restoration.source_multilevel_gradient_probe import captured_features
    reader = TinyReader(repeat=True)
    with pytest.raises(ValueError, match='exactly once'):
        captured_features(reader, torch.ones(1, 2, 16))
    assert not reader.core.id_backbone.time_down._forward_hooks
    assert not reader.core.id_backbone.t2.pool._forward_hooks


def test_capture_removes_hooks_after_reader_exception():
    from experiments.pamf_phase3_restoration.source_multilevel_gradient_probe import captured_features
    reader = TinyReader(fail=True)
    with pytest.raises(RuntimeError, match='source failure'):
        captured_features(reader, torch.ones(1, 2, 16))
    assert not reader.core.id_backbone.time_down._forward_hooks
    assert not reader.core.id_backbone.t2.pool._forward_hooks


def test_gradient_pair_reports_direction_and_zero_gradient_without_nan():
    from experiments.pamf_phase3_restoration.source_multilevel_gradient_probe import gradient_pair
    a = torch.tensor([1., 0.])
    assert gradient_pair(a, -a)['cosine'] == pytest.approx(-1.)
    assert gradient_pair(a, torch.tensor([0., 2.]))['cosine'] == pytest.approx(0.)
    assert gradient_pair(a, torch.zeros_like(a))['cosine'] is None


def test_source_pair_diagnostic_checks_input_gradients_without_model_updates():
    from experiments.pamf_phase3_restoration.source_multilevel_gradient_probe import one_record
    from experiments.pamf_phase3_restoration.objective import PairedObjective

    class Reader160(TinyReader):
        def features(self, x):
            return super().features(x).repeat(1, 80)

    class FixedRecovery(nn.Module):
        def forward(self, x, return_aux=False):
            aux = {'relative_solve_residual': torch.tensor([0.])}
            return (x, aux) if return_aux else x

    torch.manual_seed(7)
    clean = torch.randn(1, 2, 256)
    received = clean + .1 * torch.randn_like(clean)
    result = one_record(Reader160(), FixedRecovery(), PairedObjective(),
                        torch.ones(160), clean, received)
    assert set(result['losses']) == {'waveform', 'final', 'shallow', 'middle'}
    assert len(result['input_gradient_pairs']) == 6
    assert all(pair['first_norm'] > 0 and pair['second_norm'] > 0
               for pair in result['input_gradient_pairs'].values())
    assert result['posterior_residual'] == 0.
