import json

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.large_context_dura import LargeContextDura


def test_zero_initialized_output_is_original_dura():
    model = LargeContextDura(width=24, layers=2)
    x, anchor, innovation = (torch.randn(2, 2, 256) for _ in range(3))
    assert torch.equal(model(x, anchor, innovation, torch.ones(2)), anchor)


def test_large_context_model_sees_distant_samples_and_trains():
    torch.manual_seed(5)
    model = LargeContextDura(width=24, layers=2).eval()
    with torch.no_grad():
        model.out.weight.normal_(0, .02)
    received = torch.randn(1, 2, 256, requires_grad=True)
    anchor, innovation = (torch.randn(1, 2, 256) for _ in range(2))
    output = model(received, anchor, innovation, torch.tensor([.2]))
    grad = torch.autograd.grad(output[0, 0, 128], received, retain_graph=True)[0]
    assert torch.isfinite(output).all() and torch.isfinite(grad).all()
    assert grad[0, :, :16].abs().sum() > 0  # non-local interaction, not convolution alone
    loss = output.square().mean()
    loss.backward()
    assert model.out.weight.grad is not None and torch.isfinite(model.out.weight.grad).all()


def test_invalid_context_rejected():
    model = LargeContextDura(width=24, layers=2)
    x = torch.randn(2, 2, 256)
    with pytest.raises(ValueError):
        model(x, x, x, torch.tensor([1., -1.]))


def test_prediction_requires_fixed_large_model_and_legal_source(tmp_path, monkeypatch):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_context_dura import WIDTH, LAYERS, HEADS, SCALE
    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_context_dura', objective='wide_large_context',
                  source_fit_size=32768, DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  large_context_width=WIDTH, large_context_layers=LAYERS, large_context_heads=HEADS,
                  large_context_tokens=64, large_context_scale=SCALE)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    model = LargeContextDura()
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=model.state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], LargeContextDura)
    summary['large_context_width'] = WIDTH-1
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='large-context'):
        ground.load_fixed(tmp_path, 'source', 'test')
    summary['large_context_width'] = WIDTH
    summary['target_reads'] = 1
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='source-only'):
        ground.load_fixed(tmp_path, 'source', 'test')
