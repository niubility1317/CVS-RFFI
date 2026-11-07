import pytest

from experiments.phase1_dura_orthogonal import benchmark_flow_n607_graph as diagnostic


def test_wait_for_slot_respects_other_users_and_rechecks(monkeypatch, tmp_path):
    calls = []
    def output(cmd, **kwargs):
        calls.append(cmd)
        if '--query-gpu=index,uuid,memory.free' in cmd:
            return '1, GPU-A, 6000\n'
        return 'GPU-A, 1\nGPU-A, 2\nGPU-A, 3\n' if len(calls) == 2 else 'GPU-A, 1\nGPU-A, 2\n'
    sleeps = []
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', output)
    monkeypatch.setattr(diagnostic.time, 'sleep', lambda seconds: sleeps.append(seconds))
    selected_uuid = diagnostic.wait_for_slot(tmp_path/'unique', 1)
    assert selected_uuid == 'GPU-A'
    assert len(calls) == 4 and sleeps == [20]


def test_wait_for_slot_preserves_existing_output(tmp_path):
    output = tmp_path/'exists'
    output.mkdir()
    with pytest.raises(FileExistsError):
        diagnostic.wait_for_slot(output, 1)
