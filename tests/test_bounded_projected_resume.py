import json

import pytest

from experiments.pamf_phase3_restoration import bounded_projected_resume_trial as resume


def test_only_resource_stopped_parent_without_target_predictions_can_resume(tmp_path):
    parent = tmp_path / 'runs' / resume.PARENT_RUN
    folder = parent / 'rows' / resume.ARM / 'training'
    (folder / 'recovery').mkdir(parents=True)
    (folder / 'recovery' / 'E26.pt').write_bytes(b'checkpoint')
    (folder / 'smoke.json').write_text(json.dumps(dict(status='PASS', TX_labels_read=0,
        V_reads=0, OE_reads=0, target_inputs=0)), encoding='utf-8')
    (parent / 'resource_stop.json').write_text(json.dumps(dict(stage='train',
        reason='TOTAL_COMPUTE_PID_CAP_EXCEEDED', cap=3, only_own_child_signaled=True,
        other_owner_processes_signaled=False, pid=999999998)), encoding='utf-8')
    (parent / ('terminal-' + resume.ARM + '-train.json')).write_text(
        json.dumps(dict(pid=999999998, exit_code=-15)), encoding='utf-8')
    (parent / 'launch.json').write_text(json.dumps(dict(pid=999999999)), encoding='utf-8')
    assert resume.parent_training(tmp_path) == folder
    (parent / ('submitted-' + resume.ARM + '-cross_rx.json')).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='target predictions'):
        resume.parent_training(tmp_path)


def test_parent_identity_and_schedule_survive_resource_install(monkeypatch):
    parent = resume.PARENT_CONFIG.copy()
    monkeypatch.setattr(resume.original, 'RUN', resume.RUN)
    cfg = resume.configuration(resume.ARM)
    assert resume.PARENT_CONFIG == parent
    assert cfg['technical_recovery_parent'] == resume.PARENT_RUN
    assert cfg['technical_recovery_epoch'] == 26
    assert cfg['source_draws'] == parent['source_draws']
    assert len(cfg['source_draws']) == 30
    assert cfg['objective'] == parent['objective']
