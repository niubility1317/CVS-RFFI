import json
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from experiments.cvs_phase1_stack import design as d, capacity16 as q


def test_existing_source_contract_schema_is_compared_completely():
    from experiments.cvs_phase1_stack.recover import verify_contract
    expected=dict(schema='core90_game_source_roles_v1',num_classes=6,source_rxs=[1,3,4,6,8],role_ids={'L_s':['physical-1']},checkpoint_init='scratch_only',target_access_before_freeze=False)
    actual=dict(expected,native_role_comparison='EXACT_MATCH')
    verify_contract(actual,expected)
    actual['role_ids']={'L_s':['different-physical']}
    with pytest.raises(ValueError,match='role_ids'):verify_contract(actual,expected)


def test_recovery_rows_have_separate_training_origins():
    rows=d.rows()
    assert len(rows)==136
    for row in rows:
        c=d.config(row,['leo'])
        d.validate(c)
        origin=d.REUSED_R2_RUN if row['stage']=='r2' else d.RUN
        assert c['run_id']==origin
        assert c['output_root']==d.PROJECT+'/runs/'+origin+'/'+row['row_id']+'/source'


def test_completed_r2_cannot_be_retrained():
    from experiments.cvs_phase1_stack.source import train
    c=d.config(next(r for r in d.rows() if r['stage']=='r2'),['leo'])
    with pytest.raises(ValueError,match='never retrain'):
        train(c)


def test_owned_worker_completion_refills_queue(tmp_path,monkeypatch):
    done=set();started=[]
    class Child:
        def __init__(self,cmd,**kwargs):
            self.rid=Path(cmd[-1]).stem;self.pid=100+len(started);started.append(self.rid)
        def poll(self):
            done.add(self.rid);return 0
    gpu=SimpleNamespace(available_gpu=lambda active:0,occupancy=lambda active:{0:dict(pids=[],free_mb=24000)})
    monkeypatch.setitem(sys.modules,'scripts.dispatch_xuc_full',gpu)
    monkeypatch.setattr(q.subprocess,'Popen',Child)
    monkeypatch.setattr(q,'MAX_ACTIVE',1)
    monkeypatch.setattr(q,'complete',lambda d,row,kind:row['row_id'] in done)
    monkeypatch.setattr(q.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(q,'Adopted',lambda *a,**k:pytest.fail('Fresh workers must not be adopted'))
    (tmp_path/'logs/run').mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    fake=SimpleNamespace(ROOT=tmp_path,BASE=str(tmp_path),PROJECT=str(tmp_path),RUN='run',write=write,read=lambda p:json.loads(p.read_text()))
    q.queue(fake,[dict(row_id=r,config=str(tmp_path/(r+'.json'))) for r in ['first','second']],'source','r3',{})
    assert started==['first','second']
    state=fake.read(tmp_path/'queue_state.json')
    assert state['active']==state['pending']==state['failures']==[]
    assert len(fake.read(tmp_path/'launch_r3.json')['rows'])==2
