import pytest

from experiments.pamf_phase2_practical import group2_resume


def test_reuses_closed_jobs_skips_exact_failed_job_and_refuses_other_partial(tmp_path, monkeypatch):
    previous=tmp_path/'previous'; (previous/'predictions'/'done').mkdir(parents=True)
    (previous/'predictions'/'failed').mkdir()
    (previous/'predictions'/'done'/'complete.json').write_text('{}')
    jobs=[{'key':name,'output':str(tmp_path/'new'/name),'points':[(1,2)],'methods':['lora']}
          for name in ('done','failed','new')]
    monkeypatch.setattr(group2_resume, 'tasks', lambda rt: jobs)
    checked=[]
    monkeypatch.setattr(group2_resume, 'closed_prediction', lambda job, expected: checked.append((job['key'],expected)))
    monkeypatch.setattr(group2_resume, 'verify_reuse_binding', lambda rt, job: None)
    reused,deferred,pending=group2_resume.classify({}, previous, {'failed'})
    assert [v['key'] for v in reused]==['done']
    assert [v['key'] for v in deferred]==['failed']
    assert [v['key'] for v in pending]==['new']
    assert reused[0]['output']==str(previous/'predictions'/'done')
    assert checked==[('done',1)]
    (previous/'predictions'/'new').mkdir()
    with pytest.raises(ValueError,match='Unreconciled partial'):
        group2_resume.classify({}, previous, {'failed'})


def test_resume_rejects_changed_scientific_overrides():
    old={'data_root':'d', 'native_adaptation_overrides':{'dadda_sda':{'epochs':100}}, 'run_root':'old'}
    group2_resume.require_same_method(old, dict(old,run_root='new',previous_run_root='old'))
    changed=dict(old,run_root='new',native_adaptation_overrides={'dadda_sda':{'epochs':1}})
    with pytest.raises(ValueError,match='identity differs'):
        group2_resume.require_same_method(old, changed)
