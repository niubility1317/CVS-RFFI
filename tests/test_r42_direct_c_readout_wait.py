import json
from pathlib import Path
import pytest

from experiments.pamf_phase2_practical.r42_direct_c_readout_wait import wait, is_controller, controller_running


def test_wait_refuses_to_analyze_after_failed_run(tmp_path):
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'run_root': str(tmp_path)}), encoding='utf-8')
    (tmp_path / 'failed.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='worker failed'):
        wait(str(runtime), str(tmp_path / 'native.csv'), str(tmp_path / 'out'), 12345)
    assert not (tmp_path / 'out').exists()


def test_monitor_accepts_original_and_cuda_takeover_controllers_only():
    assert is_controller(['python', '-m', 'experiments.pamf_phase2_practical.r42_direct_c_dispatch', 'dispatch', '--runtime', '/run.json'], '/run.json')
    assert is_controller(['python', '-m', 'experiments.pamf_phase2_practical.r42_direct_c_takeover', '--runtime', '/run.json'], '/run.json')
    assert not is_controller(['python', '-m', 'experiments.pamf_phase2_practical.r42_direct_c_worker', '--runtime', '/run.json'], '/run.json')
    assert not is_controller(['python', '-m', 'experiments.pamf_phase2_practical.r42_direct_c_takeover', '--runtime', '/other.json'], '/run.json')


def test_controller_exit_after_completion_marker_does_not_abort_readout(tmp_path, monkeypatch):
    marker = tmp_path / 'scoring_complete.json'
    proc = tmp_path / 'cmdline'
    proc.write_bytes(b'alive')
    original = Path.read_bytes
    def disappear(path):
        if path == proc:
            marker.write_text('{}', encoding='utf-8')
            raise FileNotFoundError('controller exited during check')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', disappear)
    assert controller_running(proc, marker, '/runtime.json') is False
