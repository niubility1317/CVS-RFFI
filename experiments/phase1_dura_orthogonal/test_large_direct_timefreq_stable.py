import json

import numpy as np
import pytest
import torch

from experiments.phase1_dura_orthogonal.large_direct_timefreq_stable import StableLargeDirectTimeFreq


def test_normalized_direct_recovery_is_initially_identity_and_trains():
    torch.manual_seed(15)
    model = StableLargeDirectTimeFreq(width=16, layers=1)
    received = torch.randn(4, 2, 256)
    activations = []
    hook = model.decode.register_forward_hook(lambda module, inputs, out: activations.append(out))
    output = model(received)
    hook.remove()
    assert torch.equal(output, received)
    assert torch.count_nonzero(activations[0]) > 0
    (output - .7 * received).square().mean().backward()
    assert torch.isfinite(model.out.weight.grad).all()
    assert torch.count_nonzero(model.out.weight.grad) > 0


def test_normalization_handles_large_context():
    torch.manual_seed(16)
    model = StableLargeDirectTimeFreq(width=16, layers=1)
    context = torch.randn(3, 128, 16) * 1000
    normalized = model.transformer[-1](context)[:, :64].transpose(1, 2)
    decoded = model.decode(torch.cat((torch.randn(3, 96, 256),
        normalized.repeat_interleave(4, dim=-1)), dim=1))
    assert torch.isfinite(decoded).all()
    assert torch.count_nonzero(decoded) > decoded.numel() // 2


def test_checkpoint_requires_source_only_and_normalization(tmp_path, monkeypatch):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_direct_timefreq import WIDTH, LAYERS, HEADS, TOKENS, SCALE
    folder = tmp_path / 'runs' / 'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_direct_timefreq_stable',
                  objective='wide_large_direct_timefreq_stable', source_fit_size=32768,
                  DURA_training_reads=0, DURA_frozen=True, TX_labels_read=0,
                  F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
                  inference_received_iq_only=True, direct_timefreq_width=WIDTH,
                  direct_timefreq_layers=LAYERS, direct_timefreq_heads=HEADS,
                  direct_timefreq_tokens_per_domain=TOKENS, direct_timefreq_scale=SCALE,
                  direct_timefreq_normalization='token_layernorm_groupnorm')
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=StableLargeDirectTimeFreq().state_dict())
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder / 'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *args: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *args: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], StableLargeDirectTimeFreq)
    summary['DURA_training_reads'] = 1
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError):
        ground.load_fixed(tmp_path, 'source', 'test')
    summary['DURA_training_reads'] = 0
    summary['direct_timefreq_normalization'] = 'missing'
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='normalized direct time-frequency'):
        ground.load_fixed(tmp_path, 'source', 'test')
