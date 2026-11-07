import pytest
import torch

from experiments.phase1_dura_orthogonal.direct_circular_recovery import DirectCircularRecovery, circular_features


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)


def test_conjugate_features_corotate_under_arbitrary_global_phase():
    torch.manual_seed(6)
    x = torch.randn(3, 2, 256)
    angle = torch.tensor(.83)
    real, imag = circular_features(x)
    rotated_real, rotated_imag = circular_features(rotate(x, angle))
    torch.testing.assert_close(rotated_real, torch.cos(angle)*real-torch.sin(angle)*imag, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(rotated_imag, torch.sin(angle)*real+torch.cos(angle)*imag, rtol=2e-5, atol=2e-6)
    assert real.shape == imag.shape == (3, 4, 256)


def test_model_initial_identity_and_trained_phase_equivariance():
    torch.manual_seed(7)
    model = DirectCircularRecovery(width=4)
    x = torch.randn(3, 2, 256)
    assert torch.equal(model(x), x)
    for value in model.head.parameters():
        torch.nn.init.normal_(value, std=.1)
    angle = torch.tensor(.71)
    torch.testing.assert_close(model(rotate(x, angle)), rotate(model(x), angle), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(model(x), model(x, torch.randn_like(x), torch.randn_like(x), torch.ones(3)))
    model(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in (model.head.real, model.head.imag))


def test_zero_input_is_finite_and_bad_shape_rejected():
    model = DirectCircularRecovery(width=4)
    assert torch.isfinite(model(torch.zeros(2, 2, 256))).all()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2, 255))


def test_ground_requires_exact_source_only_circular_model(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.direct_circular_recovery import WINDOWS
    from experiments.phase1_dura_orthogonal.direct_phase_recovery import DILATIONS, RESIDUAL_SCALE

    folder=tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common=dict(commit='test',epochs=20,architecture='direct_circular_recovery',objective='wide_direct_circular',
                source_fit_size=32768,DURA_training_reads=0,DURA_frozen=True,TX_labels_read=0,
                F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,inference_received_iq_only=True,
                direct_phase_dilations=list(DILATIONS),direct_phase_residual_scale=RESIDUAL_SCALE,
                direct_circular_windows=list(WINDOWS),direct_circular_feature='local_second_moment_times_conjugate')
    summary=dict(common,status='SOURCE_TRAINED',fit_ids=32768)
    state=dict(common,schema=ground.SCHEMA,fit_ids=np.arange(32768),width=4,
               model=DirectCircularRecovery(width=4).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8')
    torch.save(state,folder/'final.pt')
    monkeypatch.setattr(ground,'frozen_dura',lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core,'teacher',lambda *a: (torch.nn.Identity(),None))
    assert isinstance(ground.load_fixed(tmp_path,'source','test')[0],DirectCircularRecovery)
    summary['direct_circular_windows']=[17,65,127]
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8')
    with pytest.raises(ValueError,match='direct circular recovery'):
        ground.load_fixed(tmp_path,'source','test')
