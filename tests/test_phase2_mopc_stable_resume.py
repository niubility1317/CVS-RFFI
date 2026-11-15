import pytest

from experiments.pamf_phase2_practical.clean_mopc_stable_resume import select_entries, retained_partial_ids


def test_successor_replaces_only_same_job_and_preserves_old_completions():
    parent=[dict(id='a',group='hybrid'),dict(id='b',group='native_cil')]
    successor=[dict(id='a',group='hybrid',new=True)]
    result=select_entries(parent,successor)
    assert [item['id'] for item in result]==['a','b'] and result[0]['new'] is True
    with pytest.raises(ValueError,match='unknown'):
        select_entries(parent,[dict(id='c',group='hybrid')])
    with pytest.raises(ValueError,match='duplicate'):
        select_entries(parent,[successor[0],successor[0]])
    with pytest.raises(ValueError,match='group'):
        select_entries(parent,[dict(id='a',group='native_cil')])


def test_retained_quarantine_requires_known_identity_and_marker(monkeypatch):
    from experiments.pamf_phase2_practical import clean_mopc_stable_resume as resume
    entries=[dict(id='a'),dict(id='b')]
    monkeypatch.setattr(resume,'numeric_quarantine_matches',lambda item:item['id']=='b')
    assert retained_partial_ids(dict(failed=[dict(id='a')],quarantined=[dict(id='b')]),entries)=={'a','b'}
    with pytest.raises(ValueError,match='quarantine'):
        retained_partial_ids(dict(quarantined=[dict(id='a')]),entries)
    with pytest.raises(ValueError,match='Unknown'):
        retained_partial_ids(dict(quarantined=[dict(id='c')]),entries)


def test_watcher_registration_failure_stops_child_and_releases_only_new_claim(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import clean_mopc_stable_resume as resume
    plan=tmp_path/'plan';plan.mkdir();reservation=tmp_path/'parent/stable_successor';reservation.parent.mkdir()
    class Process:
        pid=123;terminated=False
        def poll(self):return 0 if self.terminated else None
        def terminate(self):self.terminated=True
        def wait(self,timeout):return 0
    process=Process()
    monkeypatch.setattr(resume.subprocess,'Popen',lambda *a,**kw:process)
    real_write=resume.write
    def write(path,value):
        if path==reservation/'owner.json':raise OSError('owner failed')
        return real_write(path,value)
    monkeypatch.setattr(resume,'write',write)
    with pytest.raises(OSError,match='owner failed'):
        resume.start_watcher_and_reserve(plan,tmp_path/'run',tmp_path,tmp_path/'score',reservation)
    assert process.terminated and not reservation.exists()
