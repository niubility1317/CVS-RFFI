import torch

from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura
from experiments.phase1_dura_orthogonal.large_context_adaptive import AdaptiveLargeContextDura


def test_gain_starts_at_one_for_nonzero_correction():
    torch.manual_seed(4)
    base = StableLargeContextDura(width=16, layers=1).eval()
    with torch.no_grad():
        base.out.weight.normal_(0, .05)
    adaptive = AdaptiveLargeContextDura(width=16, layers=1).eval()
    adaptive.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(3, 2, 256)
    a = x * .7
    d = x - a
    v = torch.tensor([.01, .1, .3])
    with torch.no_grad():
        torch.testing.assert_close(adaptive(x, a, d, v), base(x, a, d, v), rtol=0, atol=0)


def test_gain_is_observable_bounded_and_learns():
    torch.manual_seed(5)
    m = AdaptiveLargeContextDura(width=16, layers=1).eval()
    with torch.no_grad():
        m.out.weight.normal_(0, .05)
    x = torch.randn(3, 2, 256)
    a = x * .7
    d = x - a
    v = torch.tensor([.01, .1, .3])
    pred = m(x, a, d, v)
    (pred - x).square().mean().backward()
    assert torch.isfinite(m.gain[-1].weight.grad).all()
    assert torch.count_nonzero(m.gain[-1].weight.grad) > 0
    with torch.no_grad():
        m.gain[-1].bias.fill_(100.)
        large = m(x, a, d, v)
        m.gain[-1].bias.fill_(-100.)
        small = m(x, a, d, v)
        stable = super(AdaptiveLargeContextDura, m).forward(x, a, d, v)
    torch.testing.assert_close(small, a, rtol=0, atol=1e-7)
    torch.testing.assert_close(large - a, 2 * (stable - a), rtol=2e-5, atol=2e-7)


def test_prediction_does_not_use_other_records():
    torch.manual_seed(6)
    m = AdaptiveLargeContextDura(width=16, layers=1).eval()
    with torch.no_grad():
        m.out.weight.normal_(0, .05)
        m.gain[-1].weight.normal_(0, .2)
    x = torch.randn(2, 2, 256); a = x * .6; d = x - a; v = torch.tensor([.1, .2])
    with torch.no_grad():
        single = m(x[:1], a[:1], d[:1], v[:1])
        together = m(x, a, d, v)[:1]
    torch.testing.assert_close(single, together, rtol=2e-5, atol=2e-7)


def test_ground_accepts_only_matching_gain_contract(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pytest
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.large_context_dura import WIDTH, LAYERS, HEADS, SCALE
    from experiments.phase1_dura_orthogonal.large_context_adaptive import GAIN_KIND
    folder = tmp_path / 'runs' / 'source'; folder.mkdir(parents=True)
    meta = dict(commit='test', epochs=20, architecture='large_context_stable',
        objective='wide_large_context_stable_adaptive', source_fit_size=32768,
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, gain_target_reads=0,
        large_context_width=WIDTH, large_context_layers=LAYERS, large_context_heads=HEADS,
        large_context_tokens=64, large_context_scale=SCALE,
        large_context_normalization='token_layernorm_groupnorm', large_context_gain=GAIN_KIND)
    summary = dict(meta, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(meta, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=WIDTH,
                 model=AdaptiveLargeContextDura().state_dict())
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder / 'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    assert isinstance(ground.load_fixed(tmp_path, 'source', 'test')[0], AdaptiveLargeContextDura)
    summary['gain_target_reads'] = 1
    (folder / 'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='observable-gain'):
        ground.load_fixed(tmp_path, 'source', 'test')
