import json
from pathlib import Path

import pytest
import signal

from experiments.pamf_phase2_practical.group2_c_backbone_fast_takeover import (
    check_parent_runtime, link_closed_predictions, worker_index, _owner_matches)
from experiments.pamf_phase2_practical.group2_c_backbone_schedule import _queue


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def test_parent_runtime_must_keep_scientific_matrix_and_inputs():
    old = {k: {'k': k} for k in ('data_root','K','N','capsules','model_seeds','b_methods','c_methods',
        'c_source_roots','hu_aux_roots_by_seed','pilot_roots','source_cache','source_contract','batch_size')}
    current = dict(old)
    check_parent_runtime(old, current)
    current['K'] = [1]
    with pytest.raises(ValueError, match='scientific'):
        check_parent_runtime(old, current)


def test_worker_index_requires_exact_module_and_old_runtime():
    cmd = ['python','-m','experiments.pamf_phase2_practical.group2_c_backbone_schedule',
           'worker','--runtime','/old/runtime.json','--index','42']
    assert worker_index(cmd, '/old/runtime.json') == 42
    assert worker_index(cmd, '/other/runtime.json') is None
    assert worker_index(cmd[:-1]+['x'], '/old/runtime.json') is None


def test_takeover_accepts_only_authorized_gpu_increase():
    old = {k: {'k': k} for k in ('data_root','K','N','capsules','model_seeds','b_methods','c_methods',
        'c_source_roots','hu_aux_roots_by_seed','pilot_roots','source_cache','source_contract','batch_size')}
    old.update(gpu_limits={str(i): 2 for i in range(8)}, minimum_free_MiB=6000)
    current = dict(old, gpu_limits={str(i): 3 if i < 7 else 2 for i in range(8)})
    check_parent_runtime(old, current)
    current['minimum_free_MiB'] = 5999
    with pytest.raises(ValueError, match='GPU policy'):
        check_parent_runtime(old, current)


def test_verified_parent_may_be_prior_takeover_controller():
    old={'release_root':'/old/release','runtime_file':'/old/release/runtime.json'}
    info={'start_ticks':19,'cwd':'/old/release','argv':['python','-m',
        'experiments.pamf_phase2_practical.group2_c_backbone_fast_takeover',
        '--runtime','/old/release/runtime.json']}
    assert _owner_matches(info,old,19)
    assert not _owner_matches(info,old,20)
    info['argv'][-1]='/different/runtime.json'
    assert not _owner_matches(info,old,19)


def test_reuse_closed_prediction_without_recomputing(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    job = {'key':'abc','b_method':'b','c_method':'c','seed':7,'capsule':'capsules/P01/MIX01/1-1',
           'points':[(5,20)],'output':str(new/'predictions/abc')}
    root=old/'predictions/abc';root.mkdir(parents=True)
    (root/'p.npz').write_bytes(b'existing prediction')
    put(root/'predictions.json', {'capsule':job['capsule'],'capsule_id':'cap','model_seed':7,
        'profile':'b','c_method':'c','rows':[{'k':5,'new_count':20,'arm':'b__c__C_SOURCE',
                                                'input_path':'P0','prediction':'p.npz'}]})
    put(root/'complete.json', {'status':'PREDICTIONS_COMPLETE','truth_read':False,'query_fit':False,
                                'rows':1,'capsule_id':'cap'})
    reused=link_closed_predictions(old,[job])
    assert reused == {0}
    assert (Path(job['output'])/'p.npz').read_bytes() == b'existing prediction'
    assert (root/'p.npz').read_bytes() == b'existing prediction'


def test_partial_parent_prediction_stops_takeover(tmp_path):
    old = tmp_path/'old'; (old/'predictions/abc').mkdir(parents=True)
    job={'key':'abc','output':str(tmp_path/'new/predictions/abc')}
    with pytest.raises((FileNotFoundError, ValueError)):
        link_closed_predictions(old,[job])


def test_queue_counts_reused_jobs_without_dispatch(tmp_path):
    rt={'run_root':str(tmp_path), 'gpu_limits':{'0':1}, 'score_processes':1, 'poll_seconds':1}
    _queue(rt,[{'key':'a'},{'key':'b'}],'worker',reused_indices={0,1})
    state=json.loads((tmp_path/'worker_state.json').read_text())
    assert state['status']=='COMPLETE' and state['completed']==state['total']==2


def test_dispatch_transfers_only_after_workers_drain_and_smoke(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import group2_c_backbone_fast_takeover as transfer
    monkeypatch.setattr(transfer.signal,'SIGSTOP',19,raising=False)
    monkeypatch.setattr(transfer.signal,'SIGCONT',18,raising=False)
    monkeypatch.setattr(transfer.signal,'SIGKILL',9,raising=False)
    oldroot, newroot = tmp_path/'old', tmp_path/'new'
    old={k:{'v':k} for k in transfer.SCIENCE_KEYS}
    old.update(gpu_limits={'0':1},minimum_free_MiB=6000,run_root=str(oldroot),
               release_root='/old/release',runtime_file='/old/release/runtime.json')
    rt=dict(old,run_root=str(newroot),release_root='/new/release',runtime_file='/new/release/runtime.json',
            parent_runtime_file=str(tmp_path/'parent.json'),parent_controller_pid=123, poll_seconds=0)
    put(tmp_path/'parent.json',old);put(oldroot/'owner.json',{'pid':123});put(oldroot/'worker_state.json',{'status':'RUNNING','completed':0})
    job={'key':'abc','b_method':'b','c_method':'c','seed':7,'capsule':'capsules/P01/MIX01/1-1',
         'points':[(5,20)],'output':str(newroot/'predictions/abc')}
    oldpred=oldroot/'predictions/abc';oldpred.mkdir(parents=True)
    (oldpred/'p.npz').write_bytes(b'fixed')
    put(oldpred/'predictions.json',{'capsule':job['capsule'],'capsule_id':'cap','model_seed':7,
        'profile':'b','c_method':'c','rows':[{'k':5,'new_count':20,'arm':'b__c__C_SOURCE','input_path':'P0','prediction':'p.npz'}]})
    put(oldpred/'complete.json',{'status':'PREDICTIONS_COMPLETE','truth_read':False,'query_fit':False,'rows':1,'capsule_id':'cap'})
    parent={'pid':123,'state':'S','start_ticks':99,'cwd':'/old/release','argv':['python','-m',
        'experiments.pamf_phase2_practical.group2_c_backbone_schedule','dispatch','--runtime','/old/release/runtime.json']}
    signals=[]; calls=[]
    def proc(pid):
        return parent if pid==123 else {'pid':456,'state':'Z','start_ticks':100}
    def kill(pid, sig):
        signals.append(sig)
        if sig==signal.SIGSTOP: parent['state']='T'
        if sig==signal.SIGCONT: parent['state']='S'
    def queue(runtime,jobs,action,**kwargs): calls.append((action,kwargs))
    monkeypatch.setattr(transfer,'validate_runtime',lambda rt:None)
    monkeypatch.setattr(transfer,'tasks',lambda rt:[job])
    monkeypatch.setattr(transfer,'_proc',proc)
    monkeypatch.setattr(transfer.os,'kill',kill)
    monkeypatch.setattr(transfer,'_children',lambda old,pid:{0:{'pid':456,'start_ticks':100}})
    monkeypatch.setattr(transfer,'_smoke',lambda rt,root:calls.append(('smoke',{})))
    monkeypatch.setattr(transfer,'_queue',queue)
    transfer.dispatch(rt)
    assert signals==[signal.SIGSTOP,signal.SIGKILL]
    assert calls==[('smoke',{}),('worker',{'reused_indices':{0}}),('score',{})]
    assert (newroot/'predictions/abc/p.npz').read_bytes()==b'fixed'
    assert json.loads((newroot/'takeover.json').read_text())['status']=='TRANSFERRED'
    assert json.loads((newroot/'predictions_complete.json').read_text())['rows']==1


def test_source_smoke_failure_resumes_verified_parent(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import group2_c_backbone_fast_takeover as transfer
    monkeypatch.setattr(transfer.signal,'SIGSTOP',19,raising=False)
    monkeypatch.setattr(transfer.signal,'SIGCONT',18,raising=False)
    oldroot,newroot=tmp_path/'old',tmp_path/'new'
    old={k:{'v':k} for k in transfer.SCIENCE_KEYS}
    old.update(gpu_limits={'0':1},minimum_free_MiB=6000,run_root=str(oldroot),
               release_root='/old/release',runtime_file='/old/release/runtime.json')
    rt=dict(old,run_root=str(newroot),parent_runtime_file=str(tmp_path/'parent.json'),parent_controller_pid=123)
    put(tmp_path/'parent.json',old);put(oldroot/'owner.json',{'pid':123});put(oldroot/'worker_state.json',{'status':'RUNNING','completed':0})
    parent={'pid':123,'state':'S','start_ticks':99,'cwd':'/old/release','argv':['python','-m',
        'experiments.pamf_phase2_practical.group2_c_backbone_schedule','dispatch','--runtime','/old/release/runtime.json']}
    signals=[]
    def kill(pid,sig):
        signals.append(sig)
        parent['state']='T' if sig==signal.SIGSTOP else 'S'
    monkeypatch.setattr(transfer,'validate_runtime',lambda rt:None)
    monkeypatch.setattr(transfer,'tasks',lambda rt:[])
    monkeypatch.setattr(transfer,'_proc',lambda pid:parent)
    monkeypatch.setattr(transfer.os,'kill',kill)
    monkeypatch.setattr(transfer,'_children',lambda old,pid:{})
    def fail(rt,root): raise ValueError('smoke failed')
    monkeypatch.setattr(transfer,'_smoke',fail)
    with pytest.raises(ValueError,match='smoke failed'): transfer.dispatch(rt)
    assert signals==[signal.SIGSTOP,signal.SIGCONT]
    assert json.loads((newroot/'takeover.json').read_text())['status']=='FAILED_PARENT_RESUMED'


def test_stale_active_worker_without_closed_output_resumes_parent(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import group2_c_backbone_fast_takeover as transfer
    monkeypatch.setattr(transfer.signal,'SIGSTOP',19,raising=False)
    monkeypatch.setattr(transfer.signal,'SIGCONT',18,raising=False)
    oldroot,newroot=tmp_path/'old',tmp_path/'new'
    old={k:{'v':k} for k in transfer.SCIENCE_KEYS}
    old.update(gpu_limits={'0':1},minimum_free_MiB=6000,run_root=str(oldroot),
               release_root='/old/release',runtime_file='/old/release/runtime.json')
    rt=dict(old,run_root=str(newroot),parent_runtime_file=str(tmp_path/'parent.json'),parent_controller_pid=123)
    put(tmp_path/'parent.json',old);put(oldroot/'owner.json',{'pid':123})
    put(oldroot/'worker_state.json',{'status':'RUNNING','completed':0,'active':[{'index':0,'pid':456}]})
    parent={'pid':123,'state':'S','start_ticks':99,'cwd':'/old/release','argv':['python','-m',
        'experiments.pamf_phase2_practical.group2_c_backbone_schedule','dispatch','--runtime','/old/release/runtime.json']}
    signals=[]
    def kill(pid,sig):
        signals.append(sig); parent['state']='T' if sig==signal.SIGSTOP else 'S'
    monkeypatch.setattr(transfer,'validate_runtime',lambda rt:None)
    monkeypatch.setattr(transfer,'tasks',lambda rt:[{'key':'abc','output':str(newroot/'predictions/abc')}])
    monkeypatch.setattr(transfer,'_proc',lambda pid:parent)
    monkeypatch.setattr(transfer.os,'kill',kill)
    monkeypatch.setattr(transfer,'_children',lambda old,pid:{})
    with pytest.raises(FileNotFoundError): transfer.dispatch(rt)
    assert signals==[signal.SIGSTOP,signal.SIGCONT]


def test_launched_reaped_failure_not_yet_in_state_resumes_parent(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import group2_c_backbone_fast_takeover as transfer
    monkeypatch.setattr(transfer.signal,'SIGSTOP',19,raising=False)
    monkeypatch.setattr(transfer.signal,'SIGCONT',18,raising=False)
    oldroot,newroot=tmp_path/'old',tmp_path/'new'
    old={k:{'v':k} for k in transfer.SCIENCE_KEYS}
    old.update(gpu_limits={'0':1},minimum_free_MiB=6000,run_root=str(oldroot),
               release_root='/old/release',runtime_file='/old/release/runtime.json')
    rt=dict(old,run_root=str(newroot),parent_runtime_file=str(tmp_path/'parent.json'),parent_controller_pid=123)
    put(tmp_path/'parent.json',old);put(oldroot/'owner.json',{'pid':123})
    put(oldroot/'worker_state.json',{'status':'RUNNING','completed':0,'active':[]})
    cmd=['python','-m','experiments.pamf_phase2_practical.group2_c_backbone_schedule','worker',
         '--runtime','/old/release/runtime.json','--index','0']
    put(oldroot/'launches/worker-00000.json',{'pid':456,'cwd':'/old/release','argv':cmd})
    parent={'pid':123,'state':'S','start_ticks':99,'cwd':'/old/release','argv':['python','-m',
        'experiments.pamf_phase2_practical.group2_c_backbone_schedule','dispatch','--runtime','/old/release/runtime.json']}
    signals=[]
    def kill(pid,sig):
        signals.append(sig); parent['state']='T' if sig==signal.SIGSTOP else 'S'
    monkeypatch.setattr(transfer,'validate_runtime',lambda rt:None)
    monkeypatch.setattr(transfer,'tasks',lambda rt:[{'key':'abc','output':str(newroot/'predictions/abc')}])
    monkeypatch.setattr(transfer,'_proc',lambda pid:parent)
    monkeypatch.setattr(transfer.os,'kill',kill)
    monkeypatch.setattr(transfer,'_children',lambda old,pid:{})
    with pytest.raises(FileNotFoundError): transfer.dispatch(rt)
    assert signals==[signal.SIGSTOP,signal.SIGCONT]
