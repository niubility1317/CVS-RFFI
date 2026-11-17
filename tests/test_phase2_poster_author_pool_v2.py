import pytest
import torch

from experiments.pamf_phase2_practical import external_additional as extra
from experiments.pamf_phase2_practical import group1, group1_source, group1_target
from experiments.pamf_phase2_practical.group2_cil_bridge import from_adapted_model


def test_author_pool_model_is_versioned_and_rejects_legacy_weights():
    author = group1_source.model_for('poster_ft_paperlast_v2', 'cpu')
    legacy = group1_source.model_for('poster_ft', 'cpu')
    assert author.network.dense1.in_features == 256
    assert legacy.network.dense1.in_features == 50
    assert 'poster_ft_paperlast_v2' not in {item['id'] for item in group1.catalog()}
    assert group1_source.recipe('poster_ft_paperlast_v2')['author_paper_exact_replication'] is False
    with pytest.raises(RuntimeError):
        author.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(ValueError):
        author(torch.randn(2, 2, 288))


def test_last_layer_adaptation_preserves_source_and_bridge_logits():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    profile = 'poster_ft_paperlast_v2'
    model = group1_source.model_for(profile, 'cpu').eval()
    x, y = torch.randn(12, 2, 256), torch.arange(6).repeat_interleave(2)
    bundle = dict(profile=profile, model=model, classes=tuple(range(6)),
                  native_state={}, resources={})
    source = {k:v.clone() for k, v in model.state_dict().items()}
    state = group1_target.fit_B({}, bundle, x, y, 23, None)
    after = state['model'].state_dict()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in source.items())
    assert all(torch.equal(v, after[k]) for k, v in source.items() if not k.startswith('network.head.'))
    assert not torch.equal(source['network.head.weight'], after['network.head.weight'])
    assert state['resources']['B_epochs'] == 10
    state['model'].eval()
    bridge = from_adapted_model(state['model'], profile, old_count=6)
    with torch.no_grad():
        assert torch.allclose(extra.native_logits(state['model'], x), bridge(x)['tx_logits'], atol=1e-6)

