import torch

from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura
from experiments.phase1_dura_orthogonal.large_context_wide_residual import WideResidualContextDura


def test_wide_context_zero_output_preserves_dura_and_has_gradient():
    torch.manual_seed(23)
    model = WideResidualContextDura(width=16, layers=1)
    x = torch.randn(3, 2, 256)
    anchor = x * .7
    direction = x - anchor
    variance = torch.full((3,), .1)
    out = model(x, anchor, direction, variance)
    assert torch.equal(out, anchor)
    (out-x).square().mean().backward()
    assert torch.isfinite(model.out.weight.grad).all()
    assert torch.count_nonzero(model.out.weight.grad) > 0


def test_wide_context_quadruples_bound_with_same_weights():
    torch.manual_seed(24)
    base = StableLargeContextDura(width=16, layers=1)
    wide = WideResidualContextDura(width=16, layers=1)
    torch.nn.init.normal_(base.out.weight, std=.1)
    wide.load_state_dict(base.state_dict(), strict=True)
    x = torch.randn(3, 2, 256);anchor=x*.7;direction=x-anchor;variance=torch.full((3,), .1)
    a=base(x,anchor,direction,variance)-anchor
    b=wide(x,anchor,direction,variance)-anchor
    assert torch.count_nonzero(a)>0
    assert torch.allclose(b,4*a,rtol=1e-5,atol=1e-6)
    assert base.residual_scale == .25 and wide.residual_scale == 1.0


def test_checkpoint_requires_correct_context_bound(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_context_dura import WIDTH, LAYERS, HEADS
    from experiments.phase1_dura_orthogonal.large_context_wide_residual import RESIDUAL_SCALE
    folder = tmp_path / 'runs' / 'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=20, architecture='large_context_stable',
        objective='wide_large_context_stable_wide', source_fit_size=32768,
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, large_context_width=WIDTH,
        large_context_layers=LAYERS, large_context_heads=HEADS, large_context_tokens=64,
        large_context_scale=RESIDUAL_SCALE, large_context_normalization='token_layernorm_groupnorm')
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=WideResidualContextDura().state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], WideResidualContextDura)
    summary['large_context_scale'] = .25
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='normalized large-context'):
        ground.load_fixed(tmp_path, 'source', 'test')
