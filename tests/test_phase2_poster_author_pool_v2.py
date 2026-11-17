import pytest
import torch
import json

from experiments.pamf_phase2_practical import external_additional as extra
from experiments.pamf_phase2_practical import group1, group1_source, group1_target
from experiments.pamf_phase2_practical.group2_cil_bridge import from_adapted_model
from experiments.poster_paper_repro import phase2_pilot


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


def test_source_scope_passes_read_only_cache_flag_without_target_handles():
    scoped = group1.source_scope(dict(augmentation_cache='/cache',
        reuse_source_augmentation=True, data_root='/target', query='/forbidden'))
    assert scoped == {'augmentation_cache':'/cache', 'reuse_source_augmentation':True}


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


def test_pilot_scores_only_after_fixed_prediction(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    rt = dict(run_root=str(root), release_root=str(tmp_path), pilot_seed=2026092701,
              pilot_capsule='capsules/P01/MIX01/1-1', reuse_source_augmentation=True)
    calls = []
    def fake_run(command, cwd, env, check):
        stage = command[-1]
        calls.append((stage, env.get('CUDA_VISIBLE_DEVICES')))
        paths = {
            'source': (root/'source'/f'{phase2_pilot.PROFILE}-s2026092701'/'group1_complete.json', 'GROUP1_SOURCE_READY'),
            'predict': (root/'predictions'/'complete.json', 'PREDICTIONS_COMPLETE'),
            'score': (root/'scores'/'complete.json', 'SCORED_COMPLETE'),
        }
        path, status = paths[stage]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'status':status}), encoding='utf8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3')
    monkeypatch.setattr(phase2_pilot.subprocess, 'run', fake_run)
    phase2_pilot.dispatch(rt, tmp_path/'runtime.json')
    assert calls == [('source', '3'), ('predict', '3'), ('score', '')]
    assert json.loads((root/'complete.json').read_text())['status'] == 'SCORED_COMPLETE'
    with pytest.raises(FileExistsError):
        phase2_pilot.dispatch(rt, tmp_path/'runtime.json')


def test_pilot_stops_if_source_completion_missing(tmp_path, monkeypatch):
    root = tmp_path/'run'
    rt = dict(run_root=str(root), release_root=str(tmp_path), pilot_seed=2026092701,
              pilot_capsule='capsules/P01/MIX01/1-1', reuse_source_augmentation=True)
    calls = []
    monkeypatch.setattr(phase2_pilot.subprocess, 'run', lambda *args, **kwargs: calls.append(args[0][-1]))
    with pytest.raises(ValueError, match='source did not produce'):
        phase2_pilot.dispatch(rt, tmp_path/'runtime.json')
    assert calls == ['source']

