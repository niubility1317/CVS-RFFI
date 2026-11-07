import pytest
import torch

from experiments.phase1_dura_orthogonal.large_context_timefreq import TimeFrequencyContextDura


def inputs():
    torch.manual_seed(17)
    x = torch.randn(3, 2, 256)
    anchor = x * 0.7
    innovation = x - anchor
    variance = torch.full((3,), 0.1)
    return x, anchor, innovation, variance


def test_zero_initialization_preserves_anchor_and_head_gradient():
    model = TimeFrequencyContextDura(width=16, layers=1)
    x, anchor, innovation, variance = inputs()
    out = model(x, anchor, innovation, variance)
    assert torch.equal(out, anchor)
    (out-x).square().mean().backward()
    assert torch.isfinite(model.out.weight.grad).all()
    assert torch.count_nonzero(model.out.weight.grad) > 0


def test_frequency_stream_changes_output_and_receives_gradients():
    model = TimeFrequencyContextDura(width=16, layers=1)
    torch.nn.init.normal_(model.out.weight, std=0.02)
    x, anchor, innovation, variance = inputs()
    normal = model(x, anchor, innovation, variance)
    handle = model.frequency_down.register_forward_hook(lambda module, args, value: torch.zeros_like(value))
    ablated = model(x, anchor, innovation, variance)
    handle.remove()
    assert not torch.allclose(normal, ablated, atol=1e-8, rtol=1e-6)
    (normal-x).square().mean().backward()
    assert torch.isfinite(model.frequency_down.weight.grad).all()
    assert torch.count_nonzero(model.frequency_down.weight.grad) > 0


def test_invalid_shapes_and_noise_rejected():
    model = TimeFrequencyContextDura(width=16, layers=1)
    x, anchor, innovation, variance = inputs()
    with pytest.raises(ValueError, match='Matching'):
        model(x[:, :, :-1], anchor, innovation, variance)
    with pytest.raises(ValueError, match='positive'):
        model(x, anchor, innovation, -variance)


def test_checkpoint_requires_matching_joint_time_frequency_metadata(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_context_dura import WIDTH, LAYERS, HEADS, SCALE
    from experiments.phase1_dura_orthogonal.large_context_timefreq import FREQUENCY_TOKENS, TIME_FREQUENCY_KIND
    folder = tmp_path / 'runs' / 'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_context_stable',
        objective='wide_large_context_stable_timefreq', source_fit_size=32768,
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, large_context_width=WIDTH,
        large_context_layers=LAYERS, large_context_heads=HEADS, large_context_tokens=64,
        large_context_scale=SCALE, large_context_normalization='token_layernorm_groupnorm',
        large_context_frequency_tokens=FREQUENCY_TOKENS, large_context_time_frequency=TIME_FREQUENCY_KIND)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=TimeFrequencyContextDura().state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], TimeFrequencyContextDura)
    summary['large_context_frequency_tokens'] = 32
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='joint time-frequency'):
        ground.load_fixed(tmp_path, 'source', 'test')
