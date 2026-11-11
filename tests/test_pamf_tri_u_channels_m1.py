import copy
import json
from pathlib import Path
import pytest
from experiments.pamf_tri_u_channels_m1 import plan


def test_single_source_contract_and_target_isolation():
    p=Path(r'E:\type10-7\local_artifacts\tri_unprotected_context.json')
    if not p.exists():pytest.skip('Local source-handle context unavailable')
    ctx=json.loads(p.read_text(encoding='utf8'));rows=plan.candidate_rows()
    assert len(rows)==12 and len({r['row_id'] for r in rows})==12
    for row in rows:
        c=plan.configuration(ctx,row,Path('/test/project'),'test');plan.validate_candidate(c)
        assert c['protocol_id']=='m1-natural' and c['source_rxs']==[1]
        bad=copy.deepcopy(c);bad['shared_prefix']=bad['shared_prefix'].replace('source_rx_1','main_split')
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['target_truth']='/forbidden'
        with pytest.raises(ValueError):plan.validate_candidate(bad)
        bad=copy.deepcopy(c);bad['channel_perturbation']['delay']=99
        with pytest.raises(ValueError):plan.validate_candidate(bad)


def test_no_gpu_admission_with_foreign_worker(monkeypatch):
    from types import SimpleNamespace
    from experiments.pamf_repl_completion import runner as capacity
    from experiments.pamf_tri_u_channels_m1.runner import admits_exclusive
    monkeypatch.setattr(capacity,'capacity',lambda gpu:({10},16000))
    monkeypatch.setattr(capacity,'processes',lambda:{10:gpu_number,20:gpu_number})
    gpu_number=7
    active={'our':(SimpleNamespace(pid=10),{'gpu':7})}
    assert not admits_exclusive(7,active,6000,base_admits=lambda *args:True)
    monkeypatch.setattr(capacity,'processes',lambda:{10:7})
    assert admits_exclusive(7,active,6000,base_admits=lambda *args:True)
    assert not admits_exclusive(7,active,6000,base_admits=lambda *args:False)
