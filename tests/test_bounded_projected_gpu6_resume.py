import json

import pytest

from experiments.pamf_phase3_restoration import bounded_projected_gpu6_resume_trial as candidate


def test_admission_uses_authorized_card_and_leaves_one_slot(monkeypatch):
    seen = {}
    def choose(**kwargs):
        seen.update(kwargs)
        return [dict(GPU=6, free_memory_MB=12288), dict(GPU=6, free_memory_MB=12287)]
    monkeypatch.setattr(candidate, 'recovery_gpu_choice', choose)
    assert candidate.admitted_gpus() == [dict(GPU=6, free_memory_MB=12288)]
    assert seen == dict(allowed_gpus=(6,), maximum_processes=4)
    assert candidate.CAP == 5


def test_unstarted_previous_queue_required(tmp_path):
    old = tmp_path / 'runs' / candidate.REPLACED_RUN
    old.mkdir(parents=True)
    (old / 'launch.json').write_text(json.dumps(dict(pid=999999999)), encoding='utf8')
    candidate.ensure_replaced_queue_inactive(tmp_path)
    (old / 'submitted-existing.json').write_text('{}', encoding='utf8')
    with pytest.raises(ValueError, match='submitted work'):
        candidate.ensure_replaced_queue_inactive(tmp_path)


def test_scientific_recipe_and_parent_remain_fixed():
    cfg = candidate.configuration(candidate.ARM)
    parent = candidate.base.PARENT_CONFIG
    assert cfg['run_id'] == candidate.RUN
    assert cfg['technical_recovery_parent'] == candidate.base.PARENT_RUN
    assert cfg['technical_recovery_epoch'] == 26
    for name in ('model_seed', 'epochs', 'updates', 'batch', 'source_draws', 'objective', 'max_relative_correction'):
        assert cfg[name] == parent[name]
    assert len(cfg['source_draws']) == 30
