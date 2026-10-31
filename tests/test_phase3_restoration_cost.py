import contextlib
import io
import runpy
import pytest
from experiments.pamf_phase3_restoration import cost_probe as m


def test_GPU_admission_counts_existing_processes_and_retains_memory_and_card_limits(monkeypatch):
    def query(command,text):
        if '--query-gpu=index,uuid,memory.free' in command:
            return '4, four, 22000\n6, six, 11999\n7, seven, 23000\n0, zero, 24000\n'
        return 'four, 1\nfour, 2\nfour, 3\nseven, 4\n'
    monkeypatch.setattr(m.subprocess,'check_output',query)
    assert m.gpu_choice()==[dict(GPU=7,free_memory_MB=23000,existing_compute_PIDs=1)]


def test_actual_CLI_exposes_only_source_cost_modes(monkeypatch):
    monkeypatch.setattr(m.sys,'argv',['probe','--help']);capture=io.StringIO()
    with contextlib.redirect_stdout(capture),pytest.raises(SystemExit) as error:
        m.main()
    assert error.value.code==0
    assert all(token in capture.getvalue() for token in ['run','worker','--root','--commit','--inputs'])
    assert 'predict' not in capture.getvalue() and 'truth' not in capture.getvalue()
