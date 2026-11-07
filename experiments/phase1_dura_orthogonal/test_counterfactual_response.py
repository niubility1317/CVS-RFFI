import torch
import pytest
import json
import numpy as np

from experiments.phase1_dura_orthogonal.counterfactual_response import (
    choose_perturbation, prepare_counterfactual, response_loss,
)


def test_selected_perturbations_cycle_and_keep_source_energy():
    torch.manual_seed(1)
    clean = torch.randn(8, 2, 256)
    selected = choose_perturbation(clean, start=0)
    later = choose_perturbation(clean, start=4)
    assert torch.equal(selected, later)
    expected = .01 * clean.square().sum((1, 2)).sqrt()
    torch.testing.assert_close(selected.abs().square().sum(1).sqrt(), expected)
    assert torch.is_complex(selected)


def test_response_loss_trains_both_paths_and_preserves_matching_response():
    original = torch.randn(3, 2, 256, requires_grad=True)
    delta = torch.randn_like(original) * .01
    changed = (original.detach() + delta * .5).requires_grad_()
    loss = response_loss(original, changed, delta)
    torch.testing.assert_close(loss, torch.tensor(.25), rtol=1e-4, atol=1e-5)
    loss.backward()
    assert original.grad.abs().sum() > 0 and changed.grad.abs().sum() > 0
    torch.testing.assert_close(response_loss(original.detach(), original.detach() + delta, delta), torch.tensor(0.), atol=1e-8, rtol=0)


def test_response_loss_rejects_mismatched_inputs():
    x = torch.zeros(2, 2, 256)
    with pytest.raises(ValueError):
        response_loss(x, x[:1], x)


def test_counterfactual_uses_fixed_source_action_and_reestimates_changed_context():
    class Dura:
        def estimate_context(self, x):
            return (None, lambda delta: 2 * delta, None)

    def changed_context(_dura, changed):
        return changed * 3, changed * 4, torch.ones(len(changed))

    torch.manual_seed(2)
    x = torch.randn(9, 2, 256)
    clean = torch.randn_like(x)
    data = (x, x, x, torch.ones(9), clean)
    changed, anchor, direction, variance, delta = prepare_counterfactual(Dura(), data, changed_context, 5)
    expected_delta = choose_perturbation(clean, 0)
    torch.testing.assert_close(torch.complex(delta[:, 0], delta[:, 1]), expected_delta)
    torch.testing.assert_close(changed, x + 2 * delta)
    torch.testing.assert_close(anchor, 3 * changed)
    torch.testing.assert_close(direction, 4 * changed)
    torch.testing.assert_close(variance, torch.ones(9))


def test_ground_loader_accepts_matching_response_metadata_and_rejects_changed_weight(tmp_path, monkeypatch):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
    from experiments.phase1_dura_orthogonal.noise_gated_dura import GATE_LOG10_LOW, GATE_LOG10_HIGH

    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=5, architecture='dura_multiscale', objective='wide_counterfactual_guard',
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, source_fit_size=32768, source_response_weight=.01,
        source_response_epsilon=.01, source_response_directions=['conjugate', 'amplitude_cubic', 'phase_slope', 'local_difference'],
        response_inference_reads=0, spectral_freq_bins=17, adaptive_filter_taps=9,
        noise_gate_low=GATE_LOG10_LOW, noise_gate_high=GATE_LOG10_HIGH)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=32,
                 model=DuraMultiscaleSpectral(width=32, taps=9).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    model, _, _ = ground.load_fixed(tmp_path, 'source', 'test')
    assert isinstance(model, DuraMultiscaleSpectral)
    summary['source_response_weight'] = .02
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='source-only response'):
        ground.load_fixed(tmp_path, 'source', 'test')
