import json

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura


def test_stable_decoder_keeps_nonzero_activations_and_head_gradients():
    torch.manual_seed(12)
    model = StableLargeContextDura(width=16, layers=1)
    received = torch.randn(4, 2, 256)
    anchor = received * 0.7
    innovation = received - anchor
    variance = torch.ones(4) * 0.1
    captured = []
    handle = model.decode.register_forward_hook(lambda module, inputs, output: captured.append(output))
    output = model(received, anchor, innovation, variance)
    handle.remove()
    assert torch.equal(output, anchor)
    assert torch.count_nonzero(captured[0]) > 0
    loss = (output - received).square().mean()
    loss.backward()
    assert torch.isfinite(model.out.weight.grad).all()
    assert torch.count_nonzero(model.out.weight.grad) > 0


def test_stable_decoder_remains_active_with_large_context():
    torch.manual_seed(13)
    model = StableLargeContextDura(width=16, layers=1)
    context = torch.randn(3, 64, 16) * 1000
    normalized = model.global_context[-1](context)
    decoded = model.decode(torch.cat((torch.randn(3, 64, 256),
        normalized.transpose(1, 2).repeat_interleave(4, dim=-1)), dim=1))
    assert torch.isfinite(decoded).all()
    assert torch.count_nonzero(decoded) > decoded.numel() // 2


def test_stable_checkpoint_requires_legal_source_and_normalization(tmp_path, monkeypatch):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_context_dura import WIDTH, LAYERS, HEADS, SCALE
    folder = tmp_path / 'runs' / 'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_context_stable',
                  objective='wide_large_context_stable', source_fit_size=32768,
                  DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  large_context_width=WIDTH, large_context_layers=LAYERS, large_context_heads=HEADS,
                  large_context_tokens=64, large_context_scale=SCALE,
                  large_context_normalization='token_layernorm_groupnorm')
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=StableLargeContextDura().state_dict())
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder / 'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], StableLargeContextDura)
    summary['large_context_normalization'] = 'missing'
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='normalized large-context'):
        ground.load_fixed(tmp_path, 'source', 'test')
