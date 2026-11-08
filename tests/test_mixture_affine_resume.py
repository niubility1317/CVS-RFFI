import json

import pytest

from experiments.pamf_phase3_restoration import mixture_affine_resume_trial as resume


def build_parent(tmp_path):
    parent = tmp_path / 'runs' / resume.PARENT_RUN
    folder = parent / 'rows' / resume.ARM / 'training'
    (folder / 'recovery').mkdir(parents=True)
    (folder / 'recovery' / 'E22.pt').write_bytes(b'checkpoint')
    (folder / 'smoke.json').write_text(json.dumps(dict(status='PASS', TX_labels_read=0,
        V_reads=0, OE_reads=0, target_inputs=0)), encoding='utf8')
    (parent / 'resource_stop.json').write_text(json.dumps(dict(stage='train',
        reason='TOTAL_COMPUTE_PID_CAP_EXCEEDED', cap=3, only_own_child_signaled=True,
        other_owner_processes_signaled=False, pid=999999998)), encoding='utf8')
    (parent / ('terminal-' + resume.ARM + '-train.json')).write_text(json.dumps(dict(
        pid=999999998, exit_code=-15, scientific_code_commit=resume.PARENT_COMMIT)), encoding='utf8')
    (parent / 'resource_controller.json').write_text(json.dumps(dict(pid=999999997)), encoding='utf8')
    (parent / 'launch.json').write_text(json.dumps(dict(pid=999999999)), encoding='utf8')
    return parent, folder


def test_only_stopped_source_parent_without_target_predictions_can_resume(tmp_path):
    parent, folder = build_parent(tmp_path)
    assert resume.parent_training(tmp_path) == folder
    (parent / ('submitted-' + resume.ARM + '-cross_rx.json')).write_text('{}', encoding='utf8')
    with pytest.raises(ValueError, match='target predictions'):
        resume.parent_training(tmp_path)


def test_refuses_foreign_process_termination_or_wrong_commit(tmp_path):
    parent, _ = build_parent(tmp_path)
    path = parent / 'resource_stop.json'
    stop = json.loads(path.read_text(encoding='utf8'))
    stop['other_owner_processes_signaled'] = True
    path.write_text(json.dumps(stop), encoding='utf8')
    with pytest.raises(ValueError, match='resource-protected stop'):
        resume.parent_training(tmp_path)


def test_recipe_and_training_budget_stay_matched():
    cfg = resume.configuration(resume.ARM)
    assert cfg['technical_recovery_parent'] == resume.PARENT_RUN
    assert cfg['technical_recovery_epoch'] == 22
    for name in ('model_seed', 'epochs', 'updates', 'batch', 'source_draws', 'objective', 'fusion'):
        assert cfg[name] == resume.PARENT_CONFIG[name]
    assert len(cfg['source_draws']) == 30


def test_gpu6_admission_preserves_headroom(monkeypatch):
    seen = {}
    def choose(**kwargs):
        seen.update(kwargs)
        return [dict(GPU=6, free_memory_MB=12288), dict(GPU=6, free_memory_MB=12287)]
    monkeypatch.setattr(resume, 'recovery_gpu_choice', choose)
    assert resume.admitted_gpus() == [dict(GPU=6, free_memory_MB=12288)]
    assert seen == dict(allowed_gpus=(6,), maximum_processes=4)
    assert resume.CAP == 5
