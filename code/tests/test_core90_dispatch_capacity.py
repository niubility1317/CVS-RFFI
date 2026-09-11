"""Capacity rejection happens before GPU access or run mutation."""
import importlib.util
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('core90_dispatch',Path(__file__).parents[1]/'scripts/dispatch_core90_game.py')
dispatch=importlib.util.module_from_spec(spec);spec.loader.exec_module(dispatch)

@pytest.mark.parametrize('capacity',[0,3,8])
def test_reject_capacity_outside_project_limit(monkeypatch,capacity):
    monkeypatch.setattr(sys,'argv',['dispatch','--matrix','unused','--release','unused','--status','unused','--max-processes-per-gpu',str(capacity)])
    with pytest.raises(ValueError,match='maximum two'):
        dispatch.main()

def test_inventory_counts_unique_foreign_processes(monkeypatch):
    responses=iter(['0, GPU-a\n1, GPU-b\n','GPU-a, 10\nGPU-a, 10\nGPU-a, 11\nGPU-b, 12\n'])
    monkeypatch.setattr(dispatch.subprocess,'check_output',lambda *a,**k:next(responses))
    counts,seen,mapping=dispatch.inventory()
    assert counts=={0:2,1:1} and len(seen)==3
