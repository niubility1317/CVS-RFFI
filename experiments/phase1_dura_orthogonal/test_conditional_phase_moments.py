import pytest
import torch

from experiments.phase1_dura_orthogonal.conditional_phase_moments import ConditionalPhaseMomentsDura, complex_moments
from experiments.phase1_dura_orthogonal.source_phase_orbit_probe import rotate


def test_moments_rotate_with_their_expected_orders_and_are_bounded():
    torch.manual_seed(7)
    x = torch.randn(5, 2, 256)
    original = complex_moments(x)
    rotated = complex_moments(rotate(x, 1))
    torch.testing.assert_close(rotated[:, :2], -original[:, :2])
    torch.testing.assert_close(rotated[:, 2:], original[:, 2:])
    assert (original.abs() <= 1.000001).all()
    assert torch.equal(complex_moments(torch.zeros_like(x)), torch.zeros(5, 4))


def test_moment_model_starts_at_dura_and_heads_learn():
    torch.manual_seed(9)
    x, anchor, innovation = (torch.randn(4, 2, 256) for _ in range(3))
    variance = torch.full((4,), .5)
    model = ConditionalPhaseMomentsDura(width=4)
    output = model(x, anchor, innovation, variance)
    torch.testing.assert_close(output, anchor, rtol=0, atol=0)
    (output-x).square().mean().backward()
    for parameter in (model.local.head.real, model.local.head.imag, model.summary[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_ground_requires_the_fixed_moments_and_unread_target(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground

    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=5, architecture='conditional_phase_moments',
                  objective='wide_conditional_phase_moments', source_fit_size=32768,
                  DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0,
                  conditional_phase_inputs='received_anchor_innovation', conditional_phase_max_radians='pi/2',
                  conditional_phase_magnitude_range=[0.9, 1.1], conditional_phase_mix='learned_per_record',
                  conditional_phase_noise_gate=True, conditional_phase_moments=[2, 4],
                  conditional_phase_moment_normalization='per_signal_power')
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=4,
                 model=ConditionalPhaseMomentsDura(width=4).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], ConditionalPhaseMomentsDura)
    summary['conditional_phase_moments'] = [2]
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='phase moment'):
        ground.load_fixed(tmp_path, 'source', 'test')
