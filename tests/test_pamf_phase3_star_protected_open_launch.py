"""Source-only scheduling, irreversible-output and independent-scorer boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.pamf_phase3_open_world import star_protected_open_launch as launch
from test_pamf_phase3_star_physical_open_train import config as old_config


def runtime(root):
    run=root/'runs'/launch.RUN;source=run.parent/launch.CACHE_RUN/'cache'
    native=old_config(root/'native')['native_config']
    def n(row,out):return dict(deepcopy(native),run_id=launch.RUN,code_commit='a'*40,row_id=row,output_root=str(out))
    cc=dict(run_id=launch.RUN,code_commit='a'*40,native_config=n('cache-s2026092701',run/'cache_native_R0'),
        L_root=str(source/'L'),OE_root=str(source/'OE'),output_root=str(run/'features'),batch_size=128)
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(run),release_root=str(root/'release'),python='python',
        launch_owner='synthetic-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
        max_concurrent_rows=3,cache_config=cc,rows={},novel_truth='private-source-truth-for-scorer-only.npz')
    for arm in launch.ARMS:
        out=run/'rows'/arm/'train'
        t=dict(run_id=launch.RUN,code_commit='a'*40,arm=arm,native_config=n(arm+'-s2026092701',out),feature_root=str(run/'features'),
            output_root=str(out),steps=2220,lr=.0001,sampling_seed=2026103017,slot_seed=2026103018)
        e=dict(run_id=launch.RUN,code_commit='a'*40,arm=arm,train_config=t,feature_root=str(run/'features'),
            cache_root=str(source),output_root=str(run/'rows'/arm/'predict'),batch_size=128)
        r['rows'][arm]=dict(train_config=t,eval_config=e)
    return r


def prepared(tmp_path):
    r=runtime(tmp_path);root=Path(r['run_root']);root.mkdir(parents=True);(root/'logs').mkdir()
    for arm in launch.ARMS:(root/'rows'/arm).mkdir(parents=True)
    p=tmp_path/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');return r,p


def test_runtime_rejects_target_truth_and_cross_role_or_output_changes(tmp_path):
    r=runtime(tmp_path);assert launch.validate(r) is r
    edits=[lambda x:x.update(target_truth='forbidden'),lambda x:x.update(unknown_K=3),lambda x:x.update(max_added_workers_per_gpu=3),
        lambda x:x['cache_config'].update(output_root=str(tmp_path/'other-cache')),
        lambda x:x['rows']['R1']['train_config'].update(feature_root=str(tmp_path/'V')),
        lambda x:x['rows']['N0']['eval_config'].update(cache_root=str(tmp_path/'target'))]
    for change in edits:
        bad=deepcopy(r);change(bad)
        with pytest.raises(ValueError):launch.validate(bad)


def test_start_existing_root_never_spawns_or_overwrites(monkeypatch,tmp_path):
    r,p=prepared(tmp_path);marker=Path(r['run_root'])/'preserve';marker.write_text('original')
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('existing root spawned'))
    with pytest.raises(FileExistsError):launch.start(p)
    assert marker.read_text()=='original'


def test_real_job_routes_truth_only_to_CPU_score_and_failure_stops_followups(monkeypatch,tmp_path):
    r,p=prepared(tmp_path);calls=[];codes=iter([0,0,0]);terminals=[]
    class Process:
        def __init__(self,argv,**kw):
            self.pid=200+len(calls);self.code=next(codes);calls.append((argv,kw))
        def wait(self):return self.code
    monkeypatch.setattr(launch.subprocess,'Popen',Process);monkeypatch.setattr(launch,'_gpu_free',lambda r:[6])
    def terminal(r,arm,stage):
        terminals.append(stage);return dict(eligible_for_target_prereg=arm=='R1',next_step='SOURCE_DIAGNOSTIC')
    monkeypatch.setattr(launch,'_terminal',terminal)
    launch.row_driver(p,'R1',6)
    assert terminals==['train','eval','score'] and len(calls)==3
    for argv,kw in calls[:2]:
        assert '--novel-truth' not in argv and kw['env']['CUDA_VISIBLE_DEVICES']=='6'
    assert calls[2][1]['env']['CUDA_VISIBLE_DEVICES']==''
    assert calls[2][0][:4]==['python','-B','-u','-m'] and '--novel-truth' in calls[2][0]
    assert '--target-truth' not in calls[2][0]
    state=launch._read(Path(r['run_root'])/'rows/R1/state.json');assert state['status']=='ANALYZED_SOURCE_ONLY'
    codes=iter([2]);calls.clear();terminals.clear()
    launch.row_driver(p,'R0',6)
    assert len(calls)==1 and not terminals
    assert launch._read(Path(r['run_root'])/'rows/R0/state.json')['status']=='FAILED_PRESERVED'
    with pytest.raises(FileExistsError):launch.row_driver(p,'R0',6)
    assert len(calls)==1


def test_terminal_checks_permissions_before_next_stage(tmp_path):
    r,_=prepared(tmp_path);folder=Path(r['run_root'])/'rows/R1/train';folder.mkdir()
    marker=dict(status='STAR_PROTECTED_SOURCE_FROZEN',run_id=launch.RUN,code_commit='a'*40,arm='R1',updates=2220,
        model_native_state_unchanged=True,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,private_truth_reads=0)
    launch._write(folder/'complete.json',marker);assert launch._terminal(r,'R1','train')==marker
    for field,value in (('updates',2219),('model_native_state_unchanged',False),('target_reads',1),('novel_validation_reads',1)):
        launch._write(folder/'complete.json',dict(marker,**{field:value}))
        with pytest.raises(ValueError):launch._terminal(r,'R1','train')


def test_cache_failure_launches_no_rows(monkeypatch,tmp_path):
    r,p=prepared(tmp_path);calls=[]
    monkeypatch.setattr(launch,'_gpu_free',lambda r:[6])
    monkeypatch.setattr(launch,'_job',lambda *args:(calls.append(args[3]),False)[1])
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('row launched after cache failure'))
    launch.driver(p)
    assert calls==['cache'] and launch._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'


def test_actual_trainer_completion_is_consumed_by_launcher_with_heavy_loop_stubbed(monkeypatch,tmp_path):
    import torch
    torch.set_num_threads(1)
    from test_pamf_phase3_star_protected_open_cache import cache_fixture
    from test_pamf_phase3_star_protected_open_train import config as train_config
    from experiments.pamf_phase3_open_world import star_protected_open_train as train
    with cache_fixture(tmp_path,monkeypatch) as (cache,cc,roles,created,events):
        r,_=prepared(tmp_path);root=Path(r['run_root'])
        cc['output_root']=str(root/'features');cache.build(cc,'cpu')
        c=train_config(tmp_path,'R1');c['feature_root']=cc['output_root']
        c['output_root']=str(root/'rows/R1/train');c['native_config']['output_root']=c['output_root']
        # Test the actual artifact producer/consumer, not 2220 expensive updates.
        # Existing cached-loop tests exercise real optimization separately.
        def loop(*args):return [dict(step=i) for i in range(1,2221)],dict(test_only_heavy_loop_stub=True)
        monkeypatch.setattr(train,'_loop',loop)
        produced=train.train(c,'cpu')
        assert produced['updates']==2220 and produced['arm']=='R1'
        assert launch._terminal(r,'R1','train')==produced


def test_actual_memory_and_two_owned_rows_per_card_policy(monkeypatch,tmp_path):
    r=runtime(tmp_path);monkeypatch.setattr(launch,'_gpu_free',lambda r:[6])
    assert launch._available_gpu(r,{'N0':dict(gpu=6)})==6
    assert launch._available_gpu(r,{'N0':dict(gpu=6),'R0':dict(gpu=6)}) is None
    monkeypatch.setattr(launch,'_gpu_free',lambda r:[6,7])
    assert launch._available_gpu(r,{'N0':dict(gpu=6)})==7


@pytest.mark.parametrize('primary_pass',[False,True])
def test_complete_pipeline_uses_only_fixed_primary(monkeypatch,tmp_path,primary_pass):
    r,p=prepared(tmp_path);calls=[]
    monkeypatch.setattr(launch,'_gpu_free',lambda r:[6,7,5])
    monkeypatch.setattr(launch,'_job',lambda *a:True);monkeypatch.setattr(launch,'_cache_terminal',lambda r:{})
    monkeypatch.setattr(launch,'_terminal',lambda r,arm,stage:dict(eligible_for_target_prereg=primary_pass,next_step='SOURCE_ONLY'))
    class Process:
        def __init__(self,argv,**kw):
            calls.append(argv);self.pid=300+len(calls);arm=argv[argv.index('--arm')+1]
            launch._write(Path(r['run_root'])/'rows'/arm/'state.json',dict(status='ANALYZED_SOURCE_ONLY'))
        def poll(self):return 0
    monkeypatch.setattr(launch.subprocess,'Popen',Process)
    launch.driver(p);state=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert len(calls)==3 and state['status']=='ANALYZED_SOURCE_ONLY' and state['all_rows_complete']
    assert state['primary']=='R1::relation_max' and state['eligible_for_target_prereg']==primary_pass
    assert state['final_phase3_complete'] is False and state['target_reads']==0
