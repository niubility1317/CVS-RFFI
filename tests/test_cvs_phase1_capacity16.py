import json
from types import SimpleNamespace
import sys
import pytest
from experiments.cvs_phase1_stack import capacity16 as c


def test_adopted_worker_is_preserved_and_completion_required(monkeypatch):
    identity=dict(pid=12,start_ticks=22,cwd='/original',argv=['python','train'])
    monkeypatch.setattr(c,'proc',lambda pid:identity)
    p=c.Adopted(identity,lambda:False);assert p.poll() is None
    monkeypatch.setattr(c,'proc',lambda pid:None)
    assert p.poll()==1
    assert c.Adopted(identity,lambda:True).poll()==0
    monkeypatch.setattr(c,'proc',lambda pid:dict(identity,start_ticks=23))
    assert p.poll()==1  # PID reuse is never adopted as the original worker.


def test_existing_config_never_overwritten(tmp_path):
    p=tmp_path/'c.json';p.write_text('{"x":1}')
    d=SimpleNamespace(read=lambda p:json.loads(p.read_text()))
    c.write_new_or_same(d,p,{'x':1})
    with pytest.raises(ValueError):c.write_new_or_same(d,p,{'x':2})
    assert json.loads(p.read_text())=={'x':1}


def test_adopt_eight_then_fill_two_per_gpu_without_duplicate_launch(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();(tmp_path/'logs/run').mkdir(parents=True)
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    d=SimpleNamespace(BASE=str(root),ROOT=tmp_path/'original_release',PROJECT=str(tmp_path),RUN='run',write=write,read=lambda p:json.loads(p.read_text()))
    jobs=[dict(row_id='row'+str(i),config='source-row'+str(i)+'.json') for i in range(16)]
    receipts=[dict(row_id='row'+str(i),pid=100+i,gpu=i) for i in range(8)]
    write(root/'launch_r2.json',dict(rows=receipts))
    identities={'row'+str(i):dict(pid=100+i,start_ticks=22,cwd='old',argv=['worker']) for i in range(8)}
    finished=False;spawned=[];snapshots=[]
    monkeypatch.setattr(c,'proc',lambda pid:None if finished else next(v for v in identities.values() if v['pid']==pid))
    monkeypatch.setattr(c,'complete',lambda *args:finished)
    class Child:
        def __init__(self,*a,**kw):self.pid=200+len(spawned);spawned.append(kw)
        def poll(self):return 0 if finished else None
    monkeypatch.setattr(c.subprocess,'Popen',Child)
    def sleep(seconds):
        nonlocal finished
        snapshots.append(d.read(root/'queue_state.json'));finished=True
    monkeypatch.setattr(c.time,'sleep',sleep)
    def occupancy(active):
        result={i:dict(pids=set(),free_mb=20000) for i in range(8)}
        for j in active.values():result[j['gpu']]['pids'].add(j['process'].pid)
        return result
    def available(active):
        caps=occupancy(active);choices=[(len(v['pids']),g) for g,v in caps.items() if len(v['pids'])<2]
        return min(choices)[1] if choices else None
    monkeypatch.setitem(sys.modules,'scripts.dispatch_xuc_full',SimpleNamespace(occupancy=occupancy,available_gpu=available))
    c.queue(d,jobs,'source','r2',identities)
    assert len(spawned)==8 and [x['env']['CUDA_VISIBLE_DEVICES'] for x in spawned]==list(map(str,range(8)))
    assert all(x['cwd']==d.ROOT for x in spawned)
    assert len(snapshots[0]['active'])==16 and snapshots[0]['max_active']==16
    final=d.read(root/'launch_r2.json')['rows'];assert final[:8]==receipts and len(final)==16


def test_adopted_failure_does_not_launch_pending(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir()
    def write(p,v):p.write_text(json.dumps(v))
    d=SimpleNamespace(BASE=str(root),ROOT=tmp_path,write=write,read=lambda p:json.loads(p.read_text()))
    write(root/'launch_r2.json',dict(rows=[dict(row_id='old',pid=1,gpu=0)]))
    monkeypatch.setattr(c,'proc',lambda pid:None);monkeypatch.setattr(c,'complete',lambda *args:False)
    monkeypatch.setitem(sys.modules,'scripts.dispatch_xuc_full',SimpleNamespace(available_gpu=lambda a:pytest.fail('Must not launch after failure'),occupancy=lambda a:{}))
    with pytest.raises(RuntimeError,match='Failed rows preserved'):
        c.queue(d,[dict(row_id='old',config='x'),dict(row_id='pending',config='y')],'source','r2',{'old':dict(pid=1,start_ticks=2)})
