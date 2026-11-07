import torch

from experiments.phase1_dura_orthogonal.large_direct_timefreq_stable import StableLargeDirectTimeFreq
from experiments.phase1_dura_orthogonal.large_direct_wide_residual import WideResidualDirectTimeFreq


def test_wide_residual_keeps_initial_identity_and_gradients():
    torch.manual_seed(19)
    model = WideResidualDirectTimeFreq(width=16, layers=1)
    x = torch.randn(3, 2, 256)
    assert torch.equal(model(x), x)
    (model(x) - x * .6).square().mean().backward()
    assert torch.isfinite(model.out.weight.grad).all()
    assert torch.count_nonzero(model.out.weight.grad) > 0


def test_wide_residual_quadruples_bound_with_same_weights():
    torch.manual_seed(20)
    ordinary = StableLargeDirectTimeFreq(width=16, layers=1)
    wide = WideResidualDirectTimeFreq(width=16, layers=1)
    wide.load_state_dict(ordinary.state_dict(), strict=True)
    torch.nn.init.normal_(ordinary.out.weight, std=.1)
    wide.load_state_dict(ordinary.state_dict(), strict=True)
    x = torch.randn(3, 2, 256)
    a = ordinary(x) - x
    b = wide(x) - x
    assert torch.count_nonzero(a) > 0
    assert torch.allclose(b, 4*a, rtol=1e-5, atol=1e-6)
    assert ordinary.residual_scale == 2.0 and wide.residual_scale == 8.0


def test_checkpoint_requires_matching_residual_bound(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_direct_timefreq import WIDTH, LAYERS, HEADS, TOKENS
    from experiments.phase1_dura_orthogonal.large_direct_wide_residual import RESIDUAL_SCALE
    folder = tmp_path / 'runs' / 'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_direct_timefreq_stable',
        objective='wide_large_direct_timefreq_stable_wide', source_fit_size=32768,
        DURA_training_reads=0, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, inference_received_iq_only=True,
        direct_timefreq_width=WIDTH, direct_timefreq_layers=LAYERS, direct_timefreq_heads=HEADS,
        direct_timefreq_tokens_per_domain=TOKENS, direct_timefreq_scale=RESIDUAL_SCALE,
        direct_timefreq_normalization='token_layernorm_groupnorm')
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=WideResidualDirectTimeFreq().state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], WideResidualDirectTimeFreq)
    summary['direct_timefreq_scale'] = 2.0
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='normalized direct time-frequency'):
        ground.load_fixed(tmp_path, 'source', 'test')
