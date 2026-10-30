import json
from pathlib import Path
import pytest
from experiments.pamf_phase3_open_world import star_metric_residual_launch as launch


def runtime(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_metric_residual_train as training,star_metric_residual_eval as evaluation
    monkeypatch.setattr(training,'validate_prepare_config',lambda c:c);monkeypatch.setattr(training,'validate_config',lambda c:c)
    monkeypatch.setattr(evaluation,'validate_config',lambda c:c)
    root=tmp_path/'run';root.mkdir();(root/'logs').mkdir();(root/'rows').mkdir()
    r=dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root=str(tmp_path/'release'),python='python',launch_owner='metric-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,max_concurrent_rows=2,novel_truth='PRIVATE_ONLY_SCORE',
        prepare_config=dict(run_id=launch.RUN,code_commit='a'*40,output_root=str(root/'features')),rows={})
    for arm in launch.ARMS:
        t=dict(run_id=launch.RUN,code_commit='a'*40,arm=arm,cache_root=str(root/'features'),output_root=str(root/'rows'/arm/'train'))
        e=dict(t,train_config=t,output_root=str(root/'rows'/arm/'predict'));r['rows'][arm]=dict(train_config=t,eval_config=e)
        (root/'rows'/arm).mkdir()
    p=tmp_path/'runtime.json';p.write_text(json.dumps(r),encoding='utf-8');monkeypatch.setattr(launch,'_gpu_free',lambda r:[0,1])
    monkeypatch.setattr(launch.time,'sleep',lambda seconds:None)
    return r,p


def marker(r,arm,stage):
    root=Path(r['run_root']);m=dict(run_id=launch.RUN,code_commit=r['code_commit'],source_only=True,target_reads=0)
    if stage=='prepare':
        m.update(status='STAR_METRIC_RESIDUAL_FEATURES_COMPLETE',counts=dict(L=6300,OE=6134),private_truth_reads=0,
            original_U_s_reads=0,G_forward_rows=0,real_F_forward_rows=0,synthetic_F_forward_rows=2);p=root/'features/complete.json'
    else:
        m.update(arm=arm,final_phase3_complete=False)
        if stage=='train':m.update(status='STAR_METRIC_RESIDUAL_SOURCE_FROZEN',updates=2220,native_updates=0,
            validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,private_truth_reads=0)
        elif stage=='eval':m.update(status='STAR_METRIC_RESIDUAL_PREDICTIONS_COMPLETE',known_count=27000,novel_count=3150,source_L_count=6300,source_OE_count=6134,
            private_truth_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,G_checkpoint_loaded=False,identity_predictions_unchanged=True,
            models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_bank_deployed=False,per_record_source_rows_deployed=0)
        else:m.update(status='STAR_METRIC_RESIDUAL_SCORES_COMPLETE',truth_last=True,full_goal_status='ACTIVE',source_gate_passed=False,eligible_for_target_prereg=False,next_step='NO_PROMOTION')
        p=root/'rows'/arm/dict(train='train',eval='predict',score='score')[stage]/'complete.json'
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m),encoding='utf-8');return m,p


def scheduler(r,monkeypatch,fail=None):
    events=[];live=[]
    def prepare(*args):marker(r,None,'prepare');events.append(('prepare','complete'));return True
    monkeypatch.setattr(launch,'_job',prepare)
    class Process:
        def __init__(self,argv,**kw):
            if 'star_metric_residual_train' in argv[4]:self.stage='train';self.arm=Path(argv[argv.index('--config')+1]).name.split('_train_config')[0]
            elif 'star_metric_residual_eval' in argv[4]:self.stage='eval';self.arm=Path(argv[argv.index('--config')+1]).name.split('_eval_config')[0]
            else:self.stage='score';self.arm=Path(argv[argv.index('--prediction-root')+1]).parent.name
            self.pid=1000+len(events);self.polls=0;self.finished=False;self.gpu=kw['env']['CUDA_VISIBLE_DEVICES'];live.append(self)
            assert sum(not p.finished and p.stage=='score' for p in live)<=1
            assert sum(not p.finished and p.stage!='score' for p in live)<=2
            assert self.gpu=='' if self.stage=='score' else self.gpu in ('0','1')
            assert ('PRIVATE_ONLY_SCORE' in argv)==(self.stage=='score');events.append((self.arm+'_'+self.stage,'start'))
        def poll(self):
            self.polls+=1
            if (self.arm=='M_LOE' and self.stage=='train' and self.polls<3) or (self.arm=='M_L' and self.stage=='score' and self.polls<3):return None
            if not self.finished:
                m,p=marker(r,self.arm,self.stage)
                if fail==(self.arm,self.stage):
                    m['native_updates' if self.stage=='train' else 'all_thresholds_frozen_before_novel']=1 if self.stage=='train' else False
                    p.write_text(json.dumps(m),encoding='utf-8')
                self.finished=True;events.append((self.arm+'_'+self.stage,'finish'))
            return 0
    monkeypatch.setattr(launch.subprocess,'Popen',Process);return events


def test_owned_config_and_resource_matrix(tmp_path,monkeypatch):
    r,_=runtime(tmp_path,monkeypatch);assert launch.validate(r) is r
    for change in (dict(target_truth='bad'),dict(max_concurrent_rows=3),dict(max_added_workers_per_gpu=3),dict(allowed_gpus=[0])):
        with pytest.raises(ValueError):launch.validate(dict(r,**change))
    r['rows']['M_L']['eval_config']['cache_root']='old-root'
    with pytest.raises(ValueError):launch.validate(r)


def test_each_frozen_row_auto_evaluates_with_one_CPU_scorer(tmp_path,monkeypatch):
    r,p=runtime(tmp_path,monkeypatch);events=scheduler(r,monkeypatch);launch.driver(p)
    assert events.index(('M_L_eval','start'))<events.index(('M_LOE_train','finish'))
    assert events.index(('M_LOE_score','start'))>events.index(('M_L_score','finish'))
    state=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_SOURCE_ONLY' and state['all_rows_complete'] and not state['eligible_for_target_prereg']
    assert state['full_goal_status']=='ACTIVE' and not state['final_phase3_complete']


@pytest.mark.parametrize('stage',['train','eval'])
def test_bad_terminal_never_scores_that_row_and_preserves_healthy_peer(tmp_path,monkeypatch,stage):
    r,p=runtime(tmp_path,monkeypatch);events=scheduler(r,monkeypatch,('M_L',stage));launch.driver(p)
    assert ('M_L_score','start') not in events and ('M_LOE_score','finish') in events
    if stage=='train':assert ('M_L_eval','start') not in events
    state=launch._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='PARTIAL_FAILED_PRESERVED' and not state['eligible_for_target_prereg']


def test_existing_output_or_pipeline_never_restarts(tmp_path,monkeypatch):
    r,p=runtime(tmp_path,monkeypatch);state=Path(r['run_root'])/'pipeline_state.json';state.write_text('old',encoding='utf-8')
    with pytest.raises(FileExistsError):launch.start(p)
    with pytest.raises(FileExistsError):launch.driver(p)
    assert state.read_text(encoding='utf-8')=='old'
