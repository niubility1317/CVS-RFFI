from pathlib import PurePosixPath as Path
import pytest
from experiments.cvs_phase1_stack import capacity16 as c

def fixture(monkeypatch):
    actual=dict(pid=123,start_ticks=456,cwd='/worker',argv=['python','-u','control.py'])
    monkeypatch.setattr(c,'proc',lambda pid:actual if pid==123 else None)
    # sys.argv excludes Python interpreter flags. The original launch receipt
    # includes them and must be checked against /proc, not normalized away.
    active=dict(pid=123,cwd='/worker',argv=['python','control.py'])
    receipt=dict(pid=123,cwd='/worker',argv=['python','-u','control.py'])
    return actual,active,receipt

def test_original_launch_receipt_retains_interpreter_flags(monkeypatch):
    actual,active,receipt=fixture(monkeypatch)
    assert c.controller_identity(active,receipt,Path('/worker'))==actual

@pytest.mark.parametrize('field,value',[('argv',['python','other.py']),('pid',124),('cwd','/other')])
def test_identity_mismatch_rejected(monkeypatch,field,value):
    _,active,receipt=fixture(monkeypatch);receipt[field]=value
    with pytest.raises(ValueError):c.controller_identity(active,receipt,Path('/worker'))
