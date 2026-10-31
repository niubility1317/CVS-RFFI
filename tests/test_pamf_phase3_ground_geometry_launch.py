"""Unlabeled source statistical fit, frozen inference, independent CPU truth."""
from pathlib import Path
import json
import pytest


def setup(root,mp):
    from experiments.pamf_phase3_open_world import ground_geometry_launch as m
    root.mkdir();(root/'logs').mkdir();mp.setattr(m._predict(),'validate_config',lambda c:c)
    ref=dict(output_root='FROZEN_GROUND',data_root=str(root.parent/'OLD_DATA'))
    e=dict(run_id=m.RUN,code_commit='a'*40,reference_config=ref,output_root=str(root/'predict'))
    r=dict(run_id=m.RUN,code_commit='a'*40,run_root=str(root),release_root=str(root.parent/'release'),python='python',launch_owner='geometry-owner',
        allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,eval_config=e,
        source_novel_truth='PRIVATE_SOURCE',clean_truth='PRIVATE_CLEAN',star_truth='PRIVATE_STAR')
    path=root.parent/(root.name+'.json');path.write_text(json.dumps(r),encoding='utf-8');mp.setattr(m,'_available',lambda r:[4]);return m,r,path


def terminal(m,r,stage):
    a=dict(run_id=m.RUN,code_commit=r['code_commit'],final_phase3_complete=False)
    if stage=='predict':a.update(status='GROUND_GEOMETRY_PREDICTIONS_COMPLETE',all_16_points_complete=True,all_four_DURA_conditions_complete=True,
        all_conditions_through_DURA=True,neural_models_frozen=True,source_subspaces_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,source_subspace_fit_count=2,nuisance_SVD_fit_count=1,source_subspace_fit_role='Us',source_subspace_fit_rows=56700,
        source_Us_reads=56700,source_U_GT_reads=0,source_L_reads=0,source_OE_reads=0,target_U_reads=0,U_ID_metadata_reads=0,
        private_truth_reads=0,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,optimizer_updates=0,
        calibration_known_count=13500,source_query_count=16650,cross_query_count=10238,real_F_forward_rows=123976,G_forward_rows=20476,
        real_SSL_forward_rows=0,real_student_forward_rows=0,new_threshold_values=12,reused_threshold_values=12,threshold_values=24,
        original_models_unchanged=True,original_default_head_unchanged=True,exact_native_classifier_coordinates=True)
    else:a.update(status='GROUND_GEOMETRY_SCORES_COMPLETE',truth_last=True,all_four_predictions_before_truth=True,all_16_points_before_truth=True,
        legacy_energy_overall_exact=True,full_goal_status='ACTIVE',new_target_DA=False,adaptation_success=None,ground_stage_success=False,
        frozen_rejection_success=False,scientific_success=False,development_gate_passed=False,next_step='NO_PROMOTION')
    out=Path(r['run_root'])/stage;out.mkdir(exist_ok=True);m._write(out/'complete.json',a);return a


def test_runtime_and_gpu_command_never_contain_private_truth(tmp_path,monkeypatch,subtests):
    m,r,_=setup(tmp_path/'run',monkeypatch);assert m.validate(r) is r
    for change in (dict(U_truth='PRIVATE'),dict(code_commit='b'*40),dict(max_added_workers_per_gpu=3),dict(minimum_free_mb=0),
                   dict(eval_config=dict(r['eval_config'],output_root='FOREIGN')),dict(train_config={})):
        with subtests.test(key=next(iter(change))),pytest.raises(ValueError):m.validate(dict(r,**change))
    p=m._command(r,'predict');assert not any(v.startswith('PRIVATE_') for v in p) and '--source-known-truth' not in p
    p=m._command(r,'score');assert '--source-known-truth' in p and 'PRIVATE_STAR' in p and any('OLD_DATA' in v for v in p)


def test_actual_job_environment_has_only_fit_predict_and_cpu_score(tmp_path,monkeypatch):
    m,r,path=setup(tmp_path/'run',monkeypatch);events=[]
    class Child:
        pid=12345
        def __init__(self,argv,**kwargs):
            stage='predict' if argv[4].endswith('.ground_geometry_run') else 'score';events.append((stage,kwargs));terminal(m,r,stage)
        def wait(self):return 0
    monkeypatch.setattr(m.subprocess,'Popen',Child);m.driver(path)
    assert [v[0] for v in events]==['predict','score'] and [v[1]['env']['CUDA_VISIBLE_DEVICES'] for v in events]==['4','']
    state=m._read(Path(r['run_root'])/'pipeline_state.json')
    assert state['status']=='ANALYZED_DEVELOPMENT' and state['primary']=='F0_domain_vim' and not state['ground_stage_success']
    assert state['adaptation_success'] is None and state['full_goal_status']=='ACTIVE'
    with pytest.raises(FileExistsError):m.start(path)
    with pytest.raises(FileExistsError):m.driver(path)


def test_wrong_labels_timing_or_twelve_points_never_connect_truth(tmp_path,monkeypatch,subtests):
    cases=(('source_U_GT_reads',1),('source_subspaces_frozen_before_CAL',False),('all_16_points_complete',False),
        ('source_Us_reads',0),('source_OE_reads',6134),('optimizer_updates',1),('exact_native_classifier_coordinates',False),('job',False))
    for key,value in cases:
        with subtests.test(key=key):
            m,r,path=setup(tmp_path/key,monkeypatch);events=[]
            def job(r,state,p,stage,argv,gpu):
                events.append(stage);a=terminal(m,r,stage)
                if key=='job':return False
                a[key]=value;m._write(Path(r['run_root'])/stage/'complete.json',a);return True
            monkeypatch.setattr(m,'_job',job);m.driver(path)
            assert events==['predict'] and not Path(r['run_root'],'score').exists()
            assert m._read(Path(r['run_root'])/'pipeline_state.json')['status']=='FAILED_PRESERVED'
