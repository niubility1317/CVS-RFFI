import json
import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.large_direct_timefreq import LargeDirectTimeFreq


def test_initial_output_is_received_and_context_is_not_read():
    model = LargeDirectTimeFreq(width=32, layers=2)
    x = torch.randn(2, 2, 256)
    assert torch.equal(model(x, torch.randn_like(x), torch.randn_like(x), torch.ones(2)), x)
    with torch.no_grad():
        model.out.weight.normal_(0, .01)
    y = model(x, torch.randn_like(x), torch.randn_like(x), torch.rand(2))
    assert torch.equal(y, model(x, None, None, None))


def test_global_and_spectral_paths_receive_gradient():
    model = LargeDirectTimeFreq(width=32, layers=2)
    with torch.no_grad():
        model.out.weight.normal_(0, .02)
    x = torch.randn(2, 2, 256, requires_grad=True)
    result = model(x)
    result.square().mean().backward()
    for p in (model.time_stem[0].weight, model.freq_stem[0].weight,
              model.transformer.layers[0].self_attn.in_proj_weight, model.out.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_invalid_received_rejected():
    model = LargeDirectTimeFreq(width=32, layers=2)
    x = torch.randn(2, 2, 256); x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(x)


def test_prediction_requires_fixed_direct_training_and_no_target(tmp_path, monkeypatch):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_direct_timefreq import WIDTH, LAYERS, HEADS, SCALE
    folder = tmp_path/'runs'/'source'; folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_direct_timefreq', objective='wide_large_direct_timefreq',
                  source_fit_size=32768, DURA_training_reads=0, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  direct_timefreq_width=WIDTH, direct_timefreq_layers=LAYERS, direct_timefreq_heads=HEADS,
                  direct_timefreq_tokens_per_domain=64, direct_timefreq_scale=SCALE,
                  inference_received_iq_only=True)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    model = LargeDirectTimeFreq()
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH, model=model.state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], LargeDirectTimeFreq)
    summary['DURA_training_reads'] = 1
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='direct time-frequency'):
        ground.load_fixed(tmp_path, 'source', 'test')
    summary['DURA_training_reads'] = 0
    summary['target_reads'] = 1
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='source-only'):
        ground.load_fixed(tmp_path, 'source', 'test')
