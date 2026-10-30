"""Launcher-only boundaries with synthetic markers and mocked subprocesses."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.pamf_phase3_open_world import star_joint_geometry_launch as launch
from experiments.pamf_phase3_open_world.star_joint_geometry_export import PARENT_COMMIT
from test_pamf_phase3_star_protected_open_eval import config as parent_config


def runtime(root):
    e=parent_config(root/'parent','R1')
    for c in (e,e['train_config'],e['train_config']['native_config']):c['code_commit']=PARENT_COMMIT
    run=root/'runs'/launch.RUN
    export=dict(run_id=launch.RUN,code_commit='b'*40,parent_eval_config=e,output_root=str(run/'features'),batch_size=128)
    return dict(run_id=launch.RUN,code_commit='b'*40,run_root=str(run),release_root=str(root/'release'),python='verified-python',
        launch_owner='synthetic-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
        export_config=export,L_public_root=str(Path(e['cache_root'])/'L'),novel_truth=str(root/'scorer-only-truth.npz'))


def prepared(root):
    r=runtime(root);run=Path(r['run_root']);(run/'logs').mkdir(parents=True);(run/'configs').mkdir()
    launch._write(run/'configs/export.json',r['export_config'])
    p=root/'runtime.json';launch._write(p,r);return r,p


def terminal(r,stage):
    m=dict(run_id=launch.RUN,code_commit=r['code_commit'],counts=dict(L=6300,novel=3150,union=9450),source_only=True,
        target_reads=0,optimizer_updates=0,final_phase3_complete=False)
    if stage=='export':
        m.update(status='STAR_JOINT_GEOMETRY_FEATURES_COMPLETE',private_truth_reads=0,model_states_unchanged=True,
            actual_adapter_checkpoint_loaded=True,native_F0_checkpoint_loaded=False,G_checkpoint_loaded=False,
            real_F_forward_rows=0,synthetic_F_forward_rows=0,synthetic_adapter_rows=2,adapter_L_feature_rows=6300,novel_feature_rows_reused=3150)
    else:
        m.update(status='STAR_JOINT_GEOMETRY_SCORES_COMPLETE',truth_last=True,all_16_neighbors_fixed_before_truth=True,
            neighbor_selection_TX_or_K_used=False,diagnostic_only=True,eligible_primary=False,full_goal_status='ACTIVE')
    return m


def test_exact_runtime_and_owned_paths_reject_private_or_cross_role_changes(tmp_path,subtests):
    r=runtime(tmp_path);assert launch.validate(r) is r
    changes=[lambda a:a.update(target_truth='forbidden'),lambda a:a.update(unknown_K=3),
        lambda a:a.update(minimum_free_mb=2048),lambda a:a.update(max_added_workers_per_gpu=3),
        lambda a:a['export_config'].update(output_root=str(tmp_path/'foreign-features')),
        lambda a:a.update(L_public_root=str(tmp_path/'parent/cache/OE')),
        lambda a:a['export_config'].update(code_commit='c'*40)]
    for i,edit in enumerate(changes):
        bad=deepcopy(r);edit(bad)
        with subtests.test(change=i),pytest.raises(ValueError):launch.validate(bad)


@pytest.mark.parametrize('stage',['export','score'])
def test_terminal_requires_identity_source_permissions_and_stage_contract(tmp_path,subtests,stage):
    r=runtime(tmp_path);folder=Path(r['run_root'])/('features' if stage=='export' else 'score');folder.mkdir(parents=True)
    good=terminal(r,stage);launch._write(folder/'complete.json',good);assert launch._terminal(r,stage)==good
    changes=dict(status='WRONG',target_reads=1,code_commit='c'*40,counts=dict(L=6300,novel=3150,union=9449))
    changes.update(dict(native_F0_checkpoint_loaded=True,model_states_unchanged=False) if stage=='export'
        else dict(all_16_neighbors_fixed_before_truth=False,eligible_primary=True))
    for key,value in changes.items():
        launch._write(folder/'complete.json',dict(good,**{key:value}))
        with subtests.test(key=key),pytest.raises(ValueError):launch._terminal(r,stage)


def test_existing_start_never_spawns_or_overwrites(tmp_path,monkeypatch):
    r,p=prepared(tmp_path);keep=Path(r['run_root'])/'preserve';keep.write_text('original',encoding='utf-8')
    monkeypatch.setattr(launch.subprocess,'Popen',lambda *a,**k:pytest.fail('Existing output spawned a controller'))
    with pytest.raises(FileExistsError):launch.start(p)
    assert keep.read_text(encoding='utf-8')=='original'


def process_stub(monkeypatch,r,events,codes=(0,0),bad_terminal=None):
    codes=iter(codes);calls=[]
    class Process:
        def __init__(self,argv,**kwargs):
            self.stage='export' if argv[4].endswith('star_joint_geometry_export') else 'score'
            self.code=next(codes);self.pid=900+len(calls);calls.append((argv,kwargs));events.append('spawn_'+self.stage)
        def wait(self):
            events.append('wait_'+self.stage)
            if self.code==0:
                m=terminal(r,self.stage)
                if self.stage==bad_terminal:m['status']='INVALID_TERMINAL'
                folder=Path(r['run_root'])/('features' if self.stage=='export' else 'score');folder.mkdir()
                launch._write(folder/'complete.json',m)
            return self.code
    monkeypatch.setattr(launch.subprocess,'Popen',Process);monkeypatch.setattr(launch,'_gpu_free',lambda r:[6])
    real=launch._terminal
    def check(r,stage):events.append('terminal_'+stage);return real(r,stage)
    monkeypatch.setattr(launch,'_terminal',check);return calls


def test_success_checks_export_then_runs_independent_CPU_score_with_truth_only_there(tmp_path,monkeypatch):
    r,p=prepared(tmp_path);events=[];calls=process_stub(monkeypatch,r,events)
    launch.driver(p)
    assert events==['spawn_export','wait_export','terminal_export','spawn_score','wait_score','terminal_score']
    export,score=calls
    assert export[1]['env']['CUDA_VISIBLE_DEVICES']=='6' and score[1]['env']['CUDA_VISIBLE_DEVICES']==''
    for argv,kwargs in calls:
        assert kwargs['cwd']==r['release_root'] and argv[:4]==['verified-python','-B','-u','-m']
        assert all(kwargs['env'][key]=='1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    assert '--novel-truth' not in export[0] and '--L-public-root' not in export[0]
    assert score[0][score[0].index('--novel-truth')+1]==r['novel_truth'] and '--target-truth' not in score[0]
    state=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_SOURCE_DIAGNOSTIC' and state['all_stages_complete']
    assert state['target_reads']==0 and state['eligible_primary'] is False and state['full_goal_status']=='ACTIVE'
    with pytest.raises(FileExistsError):launch.driver(p)
    assert len(calls)==2


@pytest.mark.parametrize('failure',['export_process','export_terminal','score_process','score_terminal'])
def test_failures_preserve_outputs_and_never_score_failed_or_unverified_export(tmp_path,monkeypatch,failure):
    r,p=prepared(tmp_path);events=[]
    codes=(2,) if failure=='export_process' else (0,3) if failure=='score_process' else (0,0)
    bad='export' if failure=='export_terminal' else 'score' if failure=='score_terminal' else None
    calls=process_stub(monkeypatch,r,events,codes,bad);launch.driver(p)
    state=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='FAILED_PRESERVED' and state['final_phase3_complete'] is False and state['error']
    if failure.startswith('export'):
        assert len(calls)==1 and 'spawn_score' not in events
    else:assert len(calls)==2 and 'terminal_export' in events
    with pytest.raises(FileExistsError):launch.driver(p)
