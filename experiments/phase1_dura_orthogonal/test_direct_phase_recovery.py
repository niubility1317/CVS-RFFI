import pytest
import torch
from experiments.phase1_dura_orthogonal.direct_phase_recovery import DirectPhaseRecovery


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)


def test_initial_identity_and_no_context_dependency():
    model = DirectPhaseRecovery(width=4)
    x = torch.randn(3, 2, 256)
    assert torch.equal(model(x), x)
    for value in model.head.parameters():
        torch.nn.init.normal_(value, std=.1)
    torch.testing.assert_close(model(x), model(x, torch.randn_like(x), torch.randn_like(x), torch.ones(3)))


def test_trained_network_is_phase_equivariant_and_both_head_parts_train():
    torch.manual_seed(4)
    model = DirectPhaseRecovery(width=4)
    for value in model.head.parameters():
        torch.nn.init.normal_(value, std=.1)
    x = torch.randn(3, 2, 256)
    angle = torch.tensor(.71)
    torch.testing.assert_close(model(rotate(x, angle)), rotate(model(x), angle), rtol=1e-5, atol=1e-5)
    model(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in (model.head.real, model.head.imag))


def test_rejects_malformed_input():
    with pytest.raises(ValueError):
        DirectPhaseRecovery(width=4)(torch.zeros(1, 2, 255))


def test_ground_requires_matching_independent_source_training(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.direct_phase_recovery import DILATIONS, RESIDUAL_SCALE

    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='direct_phase_recovery', objective='wide_direct_phase',
                  source_fit_size=32768, DURA_training_reads=0, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  inference_received_iq_only=True, direct_phase_dilations=list(DILATIONS),
                  direct_phase_residual_scale=RESIDUAL_SCALE)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=4,
                 model=DirectPhaseRecovery(width=4).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], DirectPhaseRecovery)
    summary['F_outputs_read'] = 1
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='source-only'):
        ground.load_fixed(tmp_path, 'source', 'test')
