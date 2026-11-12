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
    with pytest.raises(ValueError):validate_plan(dict(config,jobs=[entry,entry]))
    with pytest.raises(ValueError):validate_plan(dict(config,minimum_free_MiB=4000))
    with pytest.raises(ValueError):validate_plan(dict(config,phase='full'))
    dump(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='cap1',model_seed=2026092701,
                                  rows=2,truth_read=False,query_fit=False))
    assert completion_matches(entry)
    dump(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='wrong',model_seed=2026092701,
                                  rows=2,truth_read=False,query_fit=False))
    assert not completion_matches(entry)


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
