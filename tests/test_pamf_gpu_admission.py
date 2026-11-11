from types import SimpleNamespace

from experiments.pamf_physics_branch.runner import admits
from experiments.pamf_repl_completion import runner


def test_four_process_admission_counts_other_owners_and_pending_memory(monkeypatch):
    monkeypatch.setattr(runner, 'capacity', lambda gpu: ({11, 12}, 20000))
    monkeypatch.setattr(runner, 'processes', lambda: {11: 0, 12: 0, 13: 0})
    active = {'mine': (SimpleNamespace(pid=12), {'gpu': 0})}
    assert admits(0, active, 6000, max_total=4)
    assert not admits(0, active, 6000)
    monkeypatch.setattr(runner, 'capacity', lambda gpu: ({11, 12}, 11000))
    assert not admits(0, active, 6000, max_total=4)
    monkeypatch.setattr(runner, 'processes', lambda: {11: 0, 12: 0, 13: 0, 14: 0})
    assert not admits(0, active, 1000, max_total=4)
    monkeypatch.setattr(runner, 'processes', lambda: {11: 0, 12: 0, 13: 0})
    missed_own = {'mine': (SimpleNamespace(pid=14), {'gpu': 0})}
    assert not admits(0, missed_own, 1000, max_total=4)
    monkeypatch.setattr(runner, 'processes', lambda: {11: 0, 12: 0})
    assert not admits(0, missed_own, 6000, max_total=4)
