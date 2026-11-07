import torch

from experiments.phase1_dura_orthogonal.large_context_arctan import arctan_gain, ArctanLargeContextDura


def test_arctan_gain_retains_gradient_at_large_logits():
    logits = torch.tensor([-20., 0., 20.], requires_grad=True)
    gain = arctan_gain(logits)
    gain.sum().backward()
    assert torch.all(gain > 0) and torch.all(gain < 2)
    assert torch.isclose(gain[1], torch.tensor(1.))
    assert torch.isclose(logits.grad[1], torch.tensor(.5), atol=1e-6)
    assert logits.grad[0] > .001 and logits.grad[2] > .001
    assert torch.all(logits.grad > 0)


def test_zero_output_keeps_frozen_anchor_and_gain_can_receive_gradient():
    torch.manual_seed(17)
    model = ArctanLargeContextDura(width=16, layers=1)
    x = torch.randn(4, 2, 256)
    anchor = x * .7
    direction = x - anchor
    variance = torch.full((4,), .1)
    assert torch.equal(model(x, anchor, direction, variance), anchor)
    torch.nn.init.normal_(model.out.weight, std=.02)
    output = model(x, anchor, direction, variance)
    (output-x).square().mean().backward()
    assert torch.isfinite(model.gain[-1].weight.grad).all()
    assert torch.count_nonzero(model.gain[-1].weight.grad) > 0


def test_checkpoint_requires_arctan_gain_metadata(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_context_dura import WIDTH, LAYERS, HEADS, SCALE
    from experiments.phase1_dura_orthogonal.large_context_arctan import GAIN_KIND
    folder = tmp_path / 'runs' / 'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_context_stable',
        objective='wide_large_context_stable_arctan', source_fit_size=32768,
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, large_context_width=WIDTH,
        large_context_layers=LAYERS, large_context_heads=HEADS, large_context_tokens=64,
        large_context_scale=SCALE, large_context_normalization='token_layernorm_groupnorm',
        large_context_gain=GAIN_KIND, gain_target_reads=0)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=ArctanLargeContextDura().state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], ArctanLargeContextDura)
    summary['large_context_gain'] = 'observable_logenergy_2sigmoid'
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='arctan-gain'):
        ground.load_fixed(tmp_path, 'source', 'test')
