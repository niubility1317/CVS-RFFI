from pathlib import Path
from experiments.cvs_phase1_stack import capacity16 as c

def test_exit_between_stat_and_metadata_is_completion_not_identity_change(tmp_path,monkeypatch):
    folder=tmp_path/'123';folder.mkdir();(folder/'cwd').mkdir()
    (folder/'stat').write_text('123 (python) '+' '.join(['R']+['0']*18+['456']))
    (folder/'cmdline').write_bytes(b'python\0worker.py\0')
    monkeypatch.setattr(c,'Path',lambda name:tmp_path if str(name)=='/proc' else Path(name))
    old=c.proc(123);worker=c.Adopted(old,lambda:True)
    assert worker.poll() is None
    # Linux can release mm/fs while /proc/stat still reports a non-zombie task.
    (folder/'cwd').rmdir();(folder/'cmdline').write_bytes(b'')
    assert worker.poll()==0

def test_exit_without_completion_is_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'proc',lambda pid:None)
    worker=c.Adopted(dict(pid=123,start_ticks=456),lambda:False)
    assert worker.poll()==1
