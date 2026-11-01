import pytest
import signal
import subprocess


def test_merge_old_closed_predictions_overrides_older_reference():
    from experiments.pamf_phase2_practical.r42_fast_takeover import merge_refs
    old = [{'seed': 1, 'capsule': 'a', 'prediction': 'earlier', 'metrics': None},
           {'seed': 2, 'capsule': 'b', 'prediction': 'other', 'metrics': 'score'}]
    closed = [{'seed': 1, 'capsule': 'a', 'prediction': 'closed', 'metrics': None}]
    result = merge_refs(old, closed)
    assert {(r['seed'], r['capsule']): r['prediction'] for r in result} == {(1, 'a'): 'closed', (2, 'b'): 'other'}


def test_merge_rejects_duplicate_completed_task():
    from experiments.pamf_phase2_practical.r42_fast_takeover import merge_refs
    value = {'seed': 1, 'capsule': 'a', 'prediction': 'same', 'metrics': None}
    with pytest.raises(ValueError):
        merge_refs([], [value, value])


def test_scheduler_never_admits_running_or_completed_tasks():
    from experiments.pamf_phase2_practical.r42_fast_takeover import pending_tasks
    tasks = [{'key': 'a'}, {'key': 'b'}, {'key': 'c'}, {'key': 'd'}]
    assert [t['key'] for t in pending_tasks(tasks, {'a'}, {'b'}, {'c'})] == ['d']
    with pytest.raises(ValueError):
        pending_tasks(tasks, {'a'}, {'a'}, set())


def test_controller_identity_accepts_existing_takeover_only_not_smoke():
    from experiments.pamf_phase2_practical.r42_fast_takeover import controller_command_matches
    runtime = '/release/runtime.json'
    legacy = ['python','-m','experiments.pamf_phase2_practical.milkr_r42_suite','dispatch','--runtime',runtime]
    fast = ['python','-m','experiments.pamf_phase2_practical.r42_fast_takeover','--runtime',runtime]
    assert controller_command_matches(legacy,runtime)
    assert controller_command_matches(fast,runtime)
    assert not controller_command_matches(fast+['--smoke-seed','1','--smoke-output','x'],runtime)
    assert not controller_command_matches(legacy,'/different/runtime.json')


def test_retakeover_rejects_live_inherited_predecessor_workers():
    from experiments.pamf_phase2_practical.r42_fast_takeover import validate_predecessor_state
    validate_predecessor_state({'status':'RUNNING','failed':[],'predecessor_active':[]})
    with pytest.raises(ValueError, match='inherited'):
        validate_predecessor_state({'status':'RUNNING','failed':[],
                                    'predecessor_active':[{'key':'still-running'}]})
    with pytest.raises(ValueError, match='failed'):
        validate_predecessor_state({'status':'RUNNING','failed':[{'key':'bad'}],'predecessor_active':[]})


def test_parallel_score_requires_global_prediction_barrier(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover
    invoked=[]
    monkeypatch.setattr(takeover.subprocess,'Popen',lambda *a,**k: invoked.append(a))
    with pytest.raises(FileNotFoundError):
        takeover.score_queue(dict(run_root=str(tmp_path)),[dict(key='x')],2)
    assert invoked==[]


def test_parallel_score_honors_limit_and_reports_completion(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover
    root=tmp_path/'run'; (root/'logs').mkdir(parents=True)
    takeover.write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED'))
    tasks=[dict(key=str(i),seed=1,capsule='capsules/P01/LS01/rx') for i in range(3)]
    runtime=dict(run_root=str(root),runtime_file='runtime',poll_seconds=0)
    active=set(); peak=[]
    class Child:
        def __init__(self,key):
            self.key=key;self.pid=100+int(key);self.polls=0
            active.add(key);peak.append(len(active))
            output=root/'tasks'/key/'scores'/'rx'/'complete.json'
            takeover.write(output,dict(status='SCORED_COMPLETE'))
        def poll(self):
            self.polls+=1
            if self.polls==1:return None
            active.remove(self.key);return 0
    def launch(cmd,**kwargs):return Child(cmd[cmd.index('--key')+1])
    monkeypatch.setattr(takeover.subprocess,'Popen',launch)
    monkeypatch.setattr(takeover.time,'sleep',lambda _:None)
    takeover.score_queue(runtime,tasks,2)
    result=takeover.read(root/'score_state.json')
    assert max(peak)==2 and result['completed']==3 and result['status']=='COMPLETE'


def test_parallel_score_failure_drains_active_and_stops_admission(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover
    root=tmp_path/'run';(root/'logs').mkdir(parents=True)
    takeover.write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED'))
    tasks=[dict(key=str(i),seed=1,capsule='capsules/P01/LS01/rx') for i in range(3)]
    runtime=dict(run_root=str(root),runtime_file='runtime',poll_seconds=0)
    launched=[];drained=[]
    class Child:
        def __init__(self,key):
            self.key=key;self.pid=100+int(key);self.polls=0;launched.append(key)
            if key=='1':takeover.write(root/'tasks'/key/'scores'/'rx'/'complete.json',dict(status='SCORED_COMPLETE'))
        def poll(self):
            self.polls+=1
            if self.key=='1' and self.polls<3:return None
            drained.append(self.key);return 1 if self.key=='0' else 0
    monkeypatch.setattr(takeover.subprocess,'Popen',lambda cmd,**kw:Child(cmd[cmd.index('--key')+1]))
    monkeypatch.setattr(takeover.time,'sleep',lambda _:None)
    with pytest.raises(ValueError,match='Independent scoring failed'):
        takeover.score_queue(runtime,tasks,2)
    assert launched==['0','1'] and drained==['0','1']
    assert takeover.read(root/'score_state.json')['status']=='FAILED'


def test_failed_smoke_restores_paused_predecessor_without_killing_it(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover

    old = dict(release_root='old-release', runtime_file='old-runtime', allowed_gpus=[0])
    info = dict(pid=123, uid=1, cwd='old-release', start='9', state='T',
                argv=['python', '-m', takeover.suite.__package__+'.milkr_r42_suite',
                      'dispatch', '--runtime', 'old-runtime'])
    signals = []
    monkeypatch.setattr(takeover.os, 'getuid', lambda: 1, raising=False)
    monkeypatch.setattr(takeover.signal, 'SIGSTOP', 19, raising=False)
    monkeypatch.setattr(takeover.signal, 'SIGCONT', 18, raising=False)
    monkeypatch.setattr(takeover.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(takeover, 'proc', lambda pid: dict(info))
    monkeypatch.setattr(takeover, 'process_running', lambda pid,start: True, raising=False)
    monkeypatch.setattr(takeover, 'predecessor_workers', lambda *args: {})
    monkeypatch.setattr(takeover.subprocess, 'check_output', lambda *args, **kwargs: '16000')
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(takeover.subprocess, 'run', fail)
    runtime = dict(allowed_gpus=[0], minimum_free_MiB=6000, poll_seconds=1,
                   model_seeds=[1], runtime_file='new-runtime')
    with pytest.raises(subprocess.CalledProcessError):
        takeover.transfer_owner(runtime, old, 123, {}, tmp_path/'takeover.json')
    assert signals == [(123, signal.SIGSTOP), (123, signal.SIGCONT)]


def test_worker_failure_during_smoke_restores_predecessor(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover

    old = dict(release_root='old-release', runtime_file='old-runtime', allowed_gpus=[0])
    info = dict(pid=123, uid=1, cwd='old-release', start='9', state='T',
                argv=['python', '-m', takeover.suite.__package__+'.milkr_r42_suite',
                      'dispatch', '--runtime', 'old-runtime'])
    signals = []
    monkeypatch.setattr(takeover.os, 'getuid', lambda: 1, raising=False)
    monkeypatch.setattr(takeover.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    for name, value in [('SIGSTOP', 19), ('SIGCONT', 18), ('SIGKILL', 9)]:
        monkeypatch.setattr(takeover.signal, name, value, raising=False)
    monkeypatch.setattr(takeover, 'proc', lambda pid: dict(info))
    values = iter([{'a': dict(pid=124, start='10', gpu=0)}, {}])
    monkeypatch.setattr(takeover, 'predecessor_workers', lambda *args: next(values))
    monkeypatch.setattr(takeover, 'alive', lambda item: True)
    monkeypatch.setattr(takeover, 'closed_ref', lambda *args: None)
    monkeypatch.setattr(takeover.subprocess, 'check_output', lambda *args, **kwargs: '16000')
    def smoke(cmd, **kwargs):
        output = cmd[cmd.index('--smoke-output')+1]
        takeover.write(output, dict(status='PASS', seed=1, query_rows=0))
    monkeypatch.setattr(takeover.subprocess, 'run', smoke)
    runtime = dict(allowed_gpus=[0], minimum_free_MiB=6000, poll_seconds=1,
                   model_seeds=[1], runtime_file='new-runtime')
    with pytest.raises(ValueError, match='failed'):
        takeover.transfer_owner(runtime, old, 123, {'a': {'key': 'a'}}, tmp_path/'takeover.json')
    assert signals == [(123, signal.SIGSTOP), (123, signal.SIGCONT)]


def test_transfer_does_not_read_exiting_controller_environment(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover
    old=dict(release_root='old-release',runtime_file='old-runtime',allowed_gpus=[0])
    info=dict(pid=123,uid=1,cwd='old-release',start='9',state='T',
              argv=['python','-m',takeover.suite.__package__+'.milkr_r42_suite','dispatch','--runtime','old-runtime'])
    signals=[]
    monkeypatch.setattr(takeover.os,'getuid',lambda:1,raising=False)
    for name,value in [('SIGSTOP',19),('SIGCONT',18),('SIGKILL',9)]:monkeypatch.setattr(takeover.signal,name,value,raising=False)
    monkeypatch.setattr(takeover.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    def proc(pid):
        if signals and signals[-1][1]==signal.SIGKILL:raise PermissionError('exiting environ')
        return dict(info)
    monkeypatch.setattr(takeover,'proc',proc)
    monkeypatch.setattr(takeover,'process_running',lambda pid,start:False,raising=False)
    monkeypatch.setattr(takeover,'predecessor_workers',lambda *args:{})
    monkeypatch.setattr(takeover.subprocess,'check_output',lambda *args,**kwargs:'16000')
    def smoke(cmd,**kwargs):takeover.write(cmd[cmd.index('--smoke-output')+1],dict(status='PASS',seed=1,query_rows=0))
    monkeypatch.setattr(takeover.subprocess,'run',smoke)
    rt=dict(allowed_gpus=[0],minimum_free_MiB=6000,poll_seconds=1,model_seeds=[1],runtime_file='new-runtime')
    assert takeover.transfer_owner(rt,old,123,{},tmp_path/'takeover.json')=={}
    assert signals==[(123,signal.SIGSTOP),(123,signal.SIGKILL)]


def test_proc_identity_marks_linux_exiting_flag_as_not_running():
    from experiments.pamf_phase2_practical.r42_fast_takeover import running_stat_fields
    fields=['0']*20;fields[0]='R';fields[6]='0';fields[19]='42'
    assert running_stat_fields(fields,'42')
    assert not running_stat_fields(fields,'43')
    fields[6]=str(4)
    assert not running_stat_fields(fields,'42')
    fields[6]='0';fields[0]='Z'
    assert not running_stat_fields(fields,'42')


def test_adoption_rejects_disappeared_unclosed_worker(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover
    oldroot=tmp_path/'old';oldroot.mkdir()
    takeover.write(oldroot/'worker_state.json',dict(status='RUNNING',failed=[],predecessor_active=[],
        active=[dict(key='a',pid=24,gpu=0)]))
    transfer=tmp_path/'takeover.json'
    takeover.write(transfer,dict(status='PAUSED_BEFORE_TRANSFER',predecessor_controller=23,
        predecessor_start='1',inherited_workers={'a':dict(pid=24,start='2',gpu=0)}))
    monkeypatch.setattr(takeover,'proc',lambda pid:None)
    monkeypatch.setattr(takeover,'closed_ref',lambda *args:None)
    rt=dict(failed_takeover_file=str(transfer),failed_controller_pid=25,predecessor_controller_pid=23)
    with pytest.raises(ValueError,match='without complete'):
        takeover.adopt_orphans(rt,dict(run_root=str(oldroot)),{'a':dict(key='a')})


def test_adoption_keeps_owned_orphan_and_reuses_completed_stale_worker(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import r42_fast_takeover as takeover
    oldroot=tmp_path/'old';oldroot.mkdir()
    takeover.write(oldroot/'worker_state.json',dict(status='RUNNING',failed=[],predecessor_active=[],
        active=[dict(key='a',pid=24,gpu=0),dict(key='b',pid=26,gpu=0)]))
    transfer=tmp_path/'takeover.json'
    takeover.write(transfer,dict(status='PAUSED_BEFORE_TRANSFER',predecessor_controller=23,
        predecessor_start='1',inherited_workers={'a':dict(pid=24,start='2',gpu=0)}))
    argv=['python','-m',takeover.suite.__package__+'.milkr_r42_suite','worker',
          '--runtime','old-runtime','--key','a']
    info=dict(pid=24,start='2',uid=1,cwd='old-release',argv=argv,env=['CUDA_VISIBLE_DEVICES=0'])
    monkeypatch.setattr(takeover.os,'getuid',lambda:1,raising=False)
    monkeypatch.setattr(takeover,'proc',lambda pid:info if pid==24 else None)
    monkeypatch.setattr(takeover,'closed_ref',lambda old,task: {'prediction':'done'} if task['key']=='b' else None)
    rt=dict(failed_takeover_file=str(transfer),failed_controller_pid=25,predecessor_controller_pid=23)
    old=dict(run_root=str(oldroot),release_root='old-release',runtime_file='old-runtime',allowed_gpus=[0])
    assert takeover.adopt_orphans(rt,old,{'a':dict(key='a'),'b':dict(key='b')})=={'a':dict(pid=24,start='2',gpu=0)}
