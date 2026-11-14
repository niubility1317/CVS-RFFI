import json

import pytest

from experiments.pamf_phase2_practical.clean_mopc_concurrent_diagnostic import result_status


def test_nonfinite_result_is_not_mistaken_for_success(tmp_path):
    log = tmp_path/'job.log'
    log.write_text(json.dumps(dict(event='DIAGNOSTIC_RESULT', status='NONFINITE', repeat=2,
                                   query_read=False, truth_read=False, predictions_written=False))+'\n')
    assert result_status(log, 0) == 'NONFINITE'
    log.write_text(json.dumps(dict(event='DIAGNOSTIC_RESULT', status='COMPLETE', repeats=3,
                                   query_read=False, truth_read=False, predictions_written=False))+'\n')
    assert result_status(log, 0) == 'COMPLETE'
    with pytest.raises(ValueError, match='[Dd]iagnostic'):
        result_status(log, 1)


def test_resource_failure_stops_dispatch_and_drains_active(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import clean_mopc_concurrent_diagnostic as diag
    class FakeProcess:
        pid = 123
        checks = 0
        def poll(self):
            self.checks += 1
            return None if self.checks == 1 else 0
    calls = []
    def snapshot():
        calls.append(1)
        if len(calls) > 1:
            raise OSError('GPU query failed')
        return {}
    monkeypatch.setattr(diag, 'occupancy', snapshot)
    monkeypatch.setattr(diag, 'choose', lambda pending, *args, **kwargs: [(pending[0], 5)])
    monkeypatch.setattr(diag.subprocess, 'Popen', lambda *a, **kw: FakeProcess())
    monkeypatch.setattr(diag, 'result_status', lambda *a: 'COMPLETE')
    monkeypatch.setattr(diag.time, 'sleep', lambda _: None)
    root = tmp_path/'run'
    diag.run('runtime.json', 'job.json', root)
    final = json.loads((root/'state.json').read_text())
    assert final['status'] == 'NONFINITE_OR_FAILED'
    assert not final['active'] and len(final['finished']) == 1
    assert 'GPU query failed' in final['failures'][0]['reason']
