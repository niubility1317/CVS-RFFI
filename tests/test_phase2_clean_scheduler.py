import json
import pytest

from experiments.pamf_phase2_practical.clean_scheduler import choose, completion_matches, validate_plan


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)


def test_schedule_respects_owner_capacity_memory_and_priority():
    jobs=[dict(id='b',priority=1),dict(id='a',priority=0),dict(id='c',priority=0)]
    devices={0:dict(pids={999},free=20000,blocked=False),1:dict(pids=set(),free=14000,blocked=False)}
    chosen=choose(jobs,{},devices,[0,1],6000)
    assert [(j['id'],g) for j,g in chosen]==[('a',1),('c',1)]
    active={'a':dict(pid=123,gpu=1)};devices[1]['pids']={123}
    assert [(j['id'],g) for j,g in choose(jobs,active,devices,[0,1],6000)]==[('a',1)]
    devices[1]['free']=5999
    assert choose(jobs,active,devices,[0,1],6000)==[]


def test_plan_and_completion_bind_exact_clean_task(tmp_path):
    parent=tmp_path/'parent';cap=parent/'capsules/P01/MIX01/1-1';cap.mkdir(parents=True)
    dump(cap/'manifest.json',{'capsule_id':'cap1'})
    clean=tmp_path/'clean';dump(clean/'completion.json',{'status':'CLEAN_PAIRED_COMPLETE','parent_task_root':str(parent)})
    cache=tmp_path/'source_cache';cache.mkdir();phase1=tmp_path/'phase1';phase1.mkdir()
    contract=tmp_path/'contract.json';dump(contract,{})
    source=tmp_path/'source';source.mkdir();(source/'source.pt').write_bytes(b'test')
    out=tmp_path/'run/outputs/job1'
    rt=dict(clean_root=str(clean),data_root=str(parent),source_contract=str(contract),source_cache=str(cache),
            phase1_release=str(phase1),release_root=str(tmp_path),additional_sources={'tweak_m1':{'2026092701':str(source)}})
    job=dict(profile='tweak_m1',seed=2026092701,capsule='capsules/P01/MIX01/1-1',output=str(out),points=[[5,20]])
    rt_path=dump(tmp_path/'runtime.json',rt);job_path=dump(tmp_path/'job.json',job)
    entry=dict(id='one',group='native_da',runtime=rt_path,job=job_path,completion=str(out/'complete.json'))
    config=dict(run_root=str(tmp_path/'run'),jobs=[entry],max_per_gpu=2,minimum_free_MiB=6000,
                input_condition='clean_ground_pre_satellite_normalized',phase='pilot',release_root=str(tmp_path),allowed_gpus=[0,1])
    assert validate_plan(config)['jobs']==1
    assert validate_plan(dict(config,max_per_gpu=5,allowed_gpus=[1,2,3,4]))['jobs']==1
    with pytest.raises(ValueError,match='three or four'):
        validate_plan(dict(config,max_per_gpu=5))
    assert validate_plan(dict(config,max_per_gpu=5,allowed_gpus=[1,2,3]))['jobs']==1
    assert validate_plan(dict(config,max_per_gpu=7,allowed_gpus=[0,1,2,3,4],
                              worker_reserve_MiB=3000,one_per_snapshot=True))['jobs']==1
    assert validate_plan(dict(config,max_per_gpu=7,allowed_gpus=[5,6,7],
                              worker_reserve_MiB=3000,one_per_snapshot=True))['jobs']==1
    with pytest.raises(ValueError,match='three or five configured GPUs'):
        validate_plan(dict(config,max_per_gpu=7,allowed_gpus=[1,2],
                           worker_reserve_MiB=3000,one_per_snapshot=True))
    with pytest.raises(ValueError,match='staggered admission'):
        validate_plan(dict(config,max_per_gpu=7,allowed_gpus=[0,1,2,3,4]))
    with pytest.raises(ValueError):validate_plan(dict(config,jobs=[entry,entry]))
    with pytest.raises(ValueError):validate_plan(dict(config,minimum_free_MiB=4000))
    with pytest.raises(ValueError):validate_plan(dict(config,phase='full'))
    full_points=[[k,n] for k in (1,5,10,20) for n in (2,5,10,20)]
    dump(tmp_path/'job.json',dict(job,points=full_points))
    assert validate_plan(dict(config,phase='full'))['jobs']==1
    with pytest.raises(ValueError,match='Duplicate full task identity'):
        other=dict(entry,id='two',completion=str(tmp_path/'run/outputs/other/complete.json'))
        validate_plan(dict(config,phase='full',jobs=[entry,other]))
    dump(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='cap1',model_seed=2026092701,
                                  rows=2,truth_read=False,query_fit=False))
    assert completion_matches(entry)
    dump(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='wrong',model_seed=2026092701,
                                  rows=2,truth_read=False,query_fit=False))
    assert not completion_matches(entry)


def test_five_worker_limit_reserves_unallocated_memory():
    jobs=[dict(id=str(i)) for i in range(10)]
    active={str(i):dict(pid=i,gpu=1) for i in range(1,5)}
    devices={1:dict(pids={1,2,3,4},used_by_pid={1:6000,2:6000,3:6000,4:200},free=11799)}
    assert choose(jobs,active,devices,[1],6000,5)==[]
    devices[1]['free']=11800
    assert len(choose(jobs,active,devices,[1],6000,5))==1


def test_core_native_da_runs_only_when_both_baselines_are_missing(tmp_path):
    parent=tmp_path/'parent';(parent/'capsules/P01/MIX01/1-1').mkdir(parents=True)
    clean=tmp_path/'clean';dump(clean/'completion.json',{'status':'CLEAN_PAIRED_COMPLETE','parent_task_root':str(parent)})
    cache=tmp_path/'cache';cache.mkdir();phase1=tmp_path/'phase1';phase1.mkdir()
    source=tmp_path/'source';source.mkdir();(source/'source.pt').write_bytes(b'test')
    rt=dict(clean_root=str(clean),data_root=str(parent),source_contract=dump(tmp_path/'contract.json',{}),
            source_cache=str(cache),phase1_release=str(phase1),release_root=str(tmp_path),
            additional_sources={'tweak_m1':{'2026092701':str(source)}})
    cap='capsules/P01/MIX01/1-1';out=tmp_path/'run/task'
    job=dict(profile='tweak_m1',seed=2026092701,capsule=cap,output=str(out),points=[[10,5]])
    entry=dict(id='one',group='native_da',runtime=dump(tmp_path/'rt.json',rt),job=dump(tmp_path/'job.json',job),
               completion=str(out/'complete.json'))
    missing=[dict(capsule=cap,seed=2026092701,stage=stage,k=10,n=0) for stage in ('A0','B')]
    proof=dump(tmp_path/'missing.json',dict(scope='MIX01-08 K5/10 N5/10',missing={'native_da/tweak_m1':missing}))
    config=dict(run_root=str(tmp_path/'run'),jobs=[entry],max_per_gpu=2,minimum_free_MiB=6000,
                input_condition='clean_ground_pre_satellite_normalized',phase='core_native_da',release_root=str(tmp_path),
                allowed_gpus=[0,1],core_missing_file=proof)
    assert validate_plan(config)['jobs']==1
    dump(tmp_path/'missing.json',dict(scope='MIX01-08 K5/10 N5/10',missing={'native_da/tweak_m1':missing[:1]}))
    with pytest.raises(ValueError,match='already scored or unverified'):
        validate_plan(config)
    dump(tmp_path/'job.json',dict(job,points=[[10,20]]))
    with pytest.raises(ValueError,match='core'):
        validate_plan(config)


def test_seven_worker_mode_admits_one_per_snapshot_and_protects_memory():
    jobs=[dict(id=str(i)) for i in range(10)]
    active={str(i):dict(pid=i,gpu=1) for i in range(1,7)}
    devices={1:dict(pids=set(range(1,7)),used_by_pid={i:1000 for i in range(1,7)},free=19000)}
    assert len(choose(jobs,active,devices,[1],6000,7,worker_reserve=3000,one_per_snapshot=True))==1
    devices[1]['free']=17999
    assert choose(jobs,active,devices,[1],6000,7,worker_reserve=3000,one_per_snapshot=True)==[]
    devices[1]['free']=19000;devices[1]['pids'].add(999)
    assert choose(jobs,active,devices,[1],6000,7,worker_reserve=3000,one_per_snapshot=True)==[]


def test_numeric_quarantine_requires_matching_mopc_identity(tmp_path):
    from experiments.pamf_phase2_practical.clean_scheduler import numeric_quarantine_matches
    job=dict(profile='dadda_sda',capsule='capsules/P01/MIX01/1-1',seed=2026092701,methods=['mopc_hr'])
    entry=dict(group='hybrid',job=dump(tmp_path/'job.json',job),completion=str(tmp_path/'out/complete.json'))
    marker=tmp_path/'out/numerical_failure.json'
    record=dict(status='MOPC_NUMERICAL_FAILURE',profile=job['profile'],capsule=job['capsule'],seed=job['seed'],
                query_fit=False,truth_read=False)
    dump(marker,record)
    assert numeric_quarantine_matches(entry)
    dump(marker,dict(record,seed=2026092702))
    assert not numeric_quarantine_matches(entry)
    dump(marker,record)
    dump(tmp_path/'job.json',dict(job,methods=['mopc_hr','lora']))
    assert not numeric_quarantine_matches(entry)


def test_numeric_quarantine_continues_queue_without_opening_scoring(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import clean_scheduler as scheduler, runner
    monkeypatch.setattr(scheduler, 'validate_plan', lambda _: None)
    monkeypatch.setattr(runner, 'occupancy', lambda: {})
    monkeypatch.setattr(scheduler, 'choose', lambda pending, *a, **kw: [(pending[0], 5)] if pending else [])
    monkeypatch.setattr(scheduler, 'completion_matches', lambda entry: True)
    monkeypatch.setattr(scheduler.time, 'sleep', lambda _: None)
    jobs=[]
    for name in ('bad','good'):
        output=tmp_path/'run/tasks'/name
        job=dict(profile='dadda_sda',capsule='capsules/P01/MIX01/1-1',seed=2026092701,
                 methods=['mopc_hr'],output=str(output))
        jobs.append(dict(id=name,group='hybrid',job=dump(tmp_path/(name+'.json'),job),
                         runtime='runtime.json',completion=str(output/'complete.json')))
    class Process:
        pid=123
        def __init__(self, code): self.code=code
        def poll(self): return self.code
    def launch(cmd, **kwargs):
        job=json.loads(__import__('pathlib').Path(cmd[cmd.index('--job')+1]).read_text())
        output=__import__('pathlib').Path(job['output'])
        if output.name=='bad':
            dump(output/'numerical_failure.json',dict(status='MOPC_NUMERICAL_FAILURE',profile=job['profile'],
                 capsule=job['capsule'],seed=job['seed'],query_fit=False,truth_read=False))
            return Process(1)
        dump(output/'complete.json',{})
        return Process(0)
    monkeypatch.setattr(scheduler.subprocess, 'Popen', launch)
    scheduler.dispatch(dict(run_root=str(tmp_path/'run'),release_root=str(tmp_path),jobs=jobs,
        allowed_gpus=[5],minimum_free_MiB=6000,max_per_gpu=2,numerical_policy='quarantine'))
    state=json.loads((tmp_path/'run/scheduler/state.json').read_text())
    assert state['status']=='COMPLETED_WITH_QUARANTINE'
    assert state['completed']==1 and len(state['quarantined'])==1 and not state['failed']
    assert not (tmp_path/'run/scheduler/predictions_complete.json').exists()


def test_unclassified_failure_still_drains_and_does_not_score(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import clean_scheduler as scheduler, runner
    monkeypatch.setattr(scheduler, 'validate_plan', lambda _: None)
    monkeypatch.setattr(runner, 'occupancy', lambda: {})
    monkeypatch.setattr(scheduler, 'choose', lambda pending, *a, **kw: [(pending[0], 5)] if pending else [])
    monkeypatch.setattr(scheduler.time, 'sleep', lambda _: None)
    jobs=[dict(id=x,group='hybrid',job='job.json',runtime='rt.json',
               completion=str(tmp_path/'run/tasks'/x/'complete.json')) for x in ('bad','good')]
    calls=[]
    class Process:
        pid=123
        def poll(self):return 1
    monkeypatch.setattr(scheduler.subprocess, 'Popen', lambda cmd, **kw: (calls.append(cmd),Process())[1])
    with pytest.raises(RuntimeError,match='Technical failure'):
        scheduler.dispatch(dict(run_root=str(tmp_path/'run'),release_root=str(tmp_path),jobs=jobs,
            allowed_gpus=[5],minimum_free_MiB=6000,max_per_gpu=2,numerical_policy='quarantine'))
    assert len(calls)==1
    state=json.loads((tmp_path/'run/scheduler/state.json').read_text())
    assert state['status']=='FAILED' and not state['quarantined']
    assert not (tmp_path/'run/scheduler/predictions_complete.json').exists()


def test_pamf_plan_rejects_missing_restoration_cache(tmp_path):
    # The planner must not label a not-yet-produced cached run as ready.
    parent=tmp_path/'parent';parent.mkdir();clean=tmp_path/'clean'
    dump(clean/'completion.json',{'status':'CLEAN_PAIRED_COMPLETE','parent_task_root':str(parent)})
    cache=tmp_path/'cache';cache.mkdir();phase1=tmp_path/'phase1';phase1.mkdir()
    contract=tmp_path/'contract.json';dump(contract,{})
    rt=dict(clean_root=str(clean),data_root=str(parent),restored_root=str(tmp_path/'missing'),
            source_contract=str(contract),source_cache=str(cache),phase1_release=str(phase1),
            release_root=str(tmp_path),arm='R42',kernel_method='adaptive_residual_ensemble',model_seeds=[2026092701])
    job=dict(seed=2026092701,capsule='capsules/P01/MIX01/1-1',points=[[5,20]])
    entry=dict(id='pamf',group='pamf',runtime=dump(tmp_path/'rt.json',rt),job=dump(tmp_path/'job.json',job),
               completion=str(tmp_path/'run/task/predictions/1-1/complete.json'))
    config=dict(run_root=str(tmp_path/'run'),jobs=[entry],max_per_gpu=2,minimum_free_MiB=6000,phase='pilot',
                input_condition='clean_ground_pre_satellite_normalized',release_root=str(tmp_path),allowed_gpus=[0])
    with pytest.raises(FileNotFoundError):validate_plan(config)
