import json
import subprocess
import sys

import pytest

from experiments.phase1_dura_orthogonal import flow_graph_capture_probe as probe


def test_timeout_is_recorded_and_capacity_rechecked(monkeypatch, tmp_path):
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['probe', '--gpu', '1', '--out', str(out), '--commit', 'test'])
    slots = []
    monkeypatch.setattr(probe, 'wait_for_slot', lambda path, gpu: (slots.append((path, gpu)) or 'GPU-A'))
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 0, stdout='first passed')
        raise subprocess.TimeoutExpired(cmd, 120, output=b'capture stalled')
    monkeypatch.setattr(probe.subprocess, 'run', run)
    probe.main()
    result = json.loads((out/'result.json').read_text(encoding='utf8'))
    assert len(slots) == 2 and len(calls) == 2
    assert result['results']['baseline']['exit_code'] == 0
    assert result['results']['same_stream']['status'] == 'timeout'
    assert 'capture stalled' in result['results']['same_stream']['output']


def test_gpu_uuid_change_stops_and_preserves_first_result(monkeypatch, tmp_path):
    out = tmp_path/'out'
    monkeypatch.setattr(sys, 'argv', ['probe', '--gpu', '1', '--out', str(out), '--commit', 'test'])
    uuids = iter(('GPU-A', 'GPU-B'))
    monkeypatch.setattr(probe, 'wait_for_slot', lambda path, gpu: next(uuids))
    monkeypatch.setattr(probe.subprocess, 'run', lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout='first passed'))
    with pytest.raises(ValueError, match='UUID'):
        probe.main()
    result = json.loads((out/'result.json').read_text(encoding='utf8'))
    assert result['results']['baseline']['exit_code'] == 0 and 'same_stream' not in result['results']
