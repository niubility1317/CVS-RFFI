import math

import torch
import pytest

from experiments.phase1_dura_orthogonal.lag_context_dura import LagContextDura, lag_context


def rotate(x, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)


def test_lag_context_alignment_has_nonwrapping_zero_edges():
    # A signal with a constant phase step is exactly predicted by phase-aligned neighbors.
    t = torch.arange(256, dtype=torch.float32)
    z = torch.exp(1j*.031*t)
    anchor = torch.stack((z.real, z.imag))[None]
    features = lag_context(anchor)
    assert features.shape == (1, 4, 2, 256)
    torch.testing.assert_close(features, torch.zeros_like(features), rtol=0, atol=2e-6)


def test_nonperiodic_boundary_is_not_wrapped():
    x = torch.randn(2, 2, 256)
    features = lag_context(x)
    for i, lag in enumerate((20, 40)):
        assert torch.count_nonzero(features[:, 2*i, :, :lag]) == 0
        assert torch.count_nonzero(features[:, 2*i+1, :, -lag:]) == 0


def test_zero_initialization_and_joint_phase_equivariance():
    torch.manual_seed(13)
    model = LagContextDura(width=8)
    x, anchor, innovation = (torch.randn(3, 2, 256) for _ in range(3))
    variance = torch.full((3,), .2)
    assert torch.equal(model(x, anchor, innovation, variance), anchor)
    with torch.no_grad():
        model.head.real.normal_(0, .02)
        model.head.imag.normal_(0, .02)
    output = model(x, anchor, innovation, variance)
    actual = model(rotate(x, .7), rotate(anchor, .7), rotate(innovation, .7), variance)
    torch.testing.assert_close(actual, rotate(output, .7), rtol=1e-5, atol=2e-6)


def test_zero_signal_is_finite_and_heads_receive_gradients():
    model = LagContextDura(width=8)
    anchor = torch.zeros(2, 2, 256)
    assert torch.isfinite(lag_context(anchor)).all()
    x, anchor, innovation = (torch.randn(2, 2, 256) for _ in range(3))
    output = model(x, anchor, innovation, torch.full((2,), .3))
    (output-torch.randn_like(output)).square().mean().backward()
    for p in (model.head.real, model.head.imag):
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)


def test_prediction_requires_matching_source_contract(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.phase_equivariant_dura import RESIDUAL_SCALE
    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=5, architecture='lag_context_dura', objective='wide_lag_context',
                  source_fit_size=32768, DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  lag_context_lags=[20, 40], lag_context_direction='noncircular_phase_aligned_difference',
                  lag_context_inputs='received_anchor_innovation', lag_context_residual_scale=RESIDUAL_SCALE)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=4,
                 model=LagContextDura(width=4).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], LagContextDura)
    summary['lag_context_lags'] = [16, 32]
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='lag-context'):
        ground.load_fixed(tmp_path, 'source', 'test')
    summary['lag_context_lags'] = [20, 40]
    summary['target_reads'] = 1
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='source-only'):
        ground.load_fixed(tmp_path, 'source', 'test')
