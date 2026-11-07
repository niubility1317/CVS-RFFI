import pytest
import torch

from experiments.phase1_dura_orthogonal.conditional_phase_dura import ConditionalPhaseDura
from experiments.phase1_dura_orthogonal.source_phase_orbit_probe import rotate


def inputs():
    torch.manual_seed(42)
    return (torch.randn(4, 2, 256), torch.randn(4, 2, 256),
            torch.randn(4, 2, 256), torch.full((4,), .5))


def test_starts_at_frozen_anchor_and_both_heads_receive_gradients():
    x, anchor, innovation, variance = inputs()
    model = ConditionalPhaseDura(width=4)
    output = model(x, anchor, innovation, variance)
    torch.testing.assert_close(output, anchor, atol=0, rtol=0)
    (output-x).square().mean().backward()
    for parameter in (model.local.head.real, model.local.head.imag, model.summary[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_learned_absolute_phase_response_can_break_joint_equivariance():
    x, anchor, innovation, variance = inputs()
    model = ConditionalPhaseDura(width=4)
    with torch.no_grad():
        model.summary[0].weight.zero_()
        model.summary[0].bias.zero_()
        model.summary[0].weight[0, 2] = 1.0
        model.summary[-1].weight[0, 0] = 2.0
    original = model(x, anchor, innovation, variance)
    rotated = model(rotate(x, 1), rotate(anchor, 1), rotate(innovation, 1), variance)
    assert (rotated-rotate(original, 1)).abs().max() > 1e-5


def test_bad_shapes_and_nonfinite_variance_rejected():
    x, anchor, innovation, variance = inputs()
    model = ConditionalPhaseDura(width=4)
    with pytest.raises(ValueError):
        model(x[:, :, :-1], anchor, innovation, variance)
    variance[0] = float('nan')
    with pytest.raises(ValueError):
        model(x, anchor, innovation, variance)


def test_ground_enforces_frozen_source_contract(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground

    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=5, architecture='conditional_phase_dura', objective='wide_conditional_phase',
                  source_fit_size=32768, DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  conditional_phase_inputs='received_anchor_innovation', conditional_phase_max_radians='pi/2',
                  conditional_phase_magnitude_range=[0.9, 1.1], conditional_phase_mix='learned_per_record',
                  conditional_phase_noise_gate=True)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=4,
                 model=ConditionalPhaseDura(width=4).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], ConditionalPhaseDura)
    summary['target_reads'] = 1
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='source-only'):
        ground.load_fixed(tmp_path, 'source', 'test')
