import torch

from experiments.phase1_dura_orthogonal.source_forward_physics import source_channel_error


def direct_action(value, kernel, coefficients):
    h = torch.complex(kernel[:, 0], kernel[:, 1])
    main = torch.complex(coefficients[:, 0], coefficients[:, 1])
    image = torch.complex(coefficients[:, 2], coefficients[:, 3])
    idx = torch.cat((torch.arange(63, 0, -1), torch.arange(256)))
    extended = value[:, idx]
    filtered = sum(h[:, j:j+1] * extended[:, 63-j:63-j+256] for j in range(64))
    return main * filtered + image * filtered.conj()


def test_channel_error_matches_real_linear_reference_and_gradients():
    torch.manual_seed(17)
    output = torch.randn(3, 2, 256, requires_grad=True)
    clean = torch.randn(3, 2, 256, requires_grad=True)
    kernel = torch.randn(3, 2, 64, requires_grad=True)
    coeff = torch.randn(3, 4, 256, requires_grad=True)
    actual = source_channel_error(output, clean, kernel, coeff)
    z = torch.complex(output[:, 0], output[:, 1])
    y = torch.complex(clean[:, 0].detach(), clean[:, 1].detach())
    h = kernel.detach(); c = coeff.detach()
    ref = (direct_action(z-y, h, c).abs().square().mean(1) /
           direct_action(y, h, c).abs().square().mean(1).clamp_min(1e-8)).mean()
    torch.testing.assert_close(actual, ref, rtol=1e-5, atol=1e-6)
    actual.backward()
    assert torch.isfinite(output.grad).all() and torch.count_nonzero(output.grad) > 0
    assert clean.grad is None and kernel.grad is None and coeff.grad is None


def test_perfect_source_recovery_has_zero_channel_loss():
    x = torch.randn(2, 2, 256)
    h = torch.randn(2, 2, 64); c = torch.randn(2, 4, 256)
    assert source_channel_error(x, x, h, c).item() == 0.


def test_ground_requires_source_only_physical_loss(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_direct_timefreq import WIDTH, LAYERS, HEADS, TOKENS, SCALE
    from experiments.phase1_dura_orthogonal.large_direct_timefreq_stable import StableLargeDirectTimeFreq
    from experiments.phase1_dura_orthogonal.source_forward_physics import WEIGHT
    folder = tmp_path / 'runs' / 'source'; folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_direct_timefreq_stable',
        objective='wide_large_direct_timefreq_stable_physics', source_fit_size=32768,
        DURA_training_reads=0, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, inference_received_iq_only=True,
        direct_timefreq_width=WIDTH, direct_timefreq_layers=LAYERS, direct_timefreq_heads=HEADS,
        direct_timefreq_tokens_per_domain=TOKENS, direct_timefreq_scale=SCALE,
        direct_timefreq_normalization='token_layernorm_groupnorm',
        source_forward_physics_weight=WEIGHT, source_forward_physics_losses=[.1]*20,
        source_simulator_action_used=True,
        simulator_action_inference_reads=0)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=StableLargeDirectTimeFreq().state_dict())
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder / 'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], StableLargeDirectTimeFreq)
    summary['simulator_action_inference_reads'] = 1
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='forward-physics'):
        ground.load_fixed(tmp_path, 'source', 'test')
