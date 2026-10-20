import copy
import numpy as np
import pytest

from tools.analyze_pamf_r17_task_results import classification_record,summary_metrics,compare,training_metrics,completed_metrics
from experiments.pamf_r17_task import design as d
from experiments.pamf_r17_task.evaluation import select_source


def record(matrix,**fields):
    matrix=np.asarray(matrix,np.int64);n=int(matrix.sum());correct=int(matrix.trace())
    precision=np.divide(matrix.diagonal(),matrix.sum(0),out=np.zeros(6),where=matrix.sum(0)>0)
    recall=np.divide(matrix.diagonal(),matrix.sum(1),out=np.zeros(6),where=matrix.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(6),where=precision+recall>0)
    return dict(confusion=matrix.tolist(),count=n,accuracy=100*correct/n,macro_f1=100*float(f1.mean()),
                baseline_accuracy=100.,repaired=0,damaged=n-correct,net_repaired=correct-n,readout='restored',**fields)


def summary(role='source',errors=180):
    n=1800 if role=='source' else 168000;records=[]
    for view in d.VIEWS:
        matrix=np.eye(6,dtype=np.int64)*(n//6);matrix[0,0]-=errors;matrix[0,1]+=errors
        records.append(record(matrix,view=view,dimension='overall',value='all'))
        for label in range(6):
            part=np.zeros_like(matrix);part[label]=matrix[label]
            records.append(record(part,view=view,dimension='TX',value=str(label)))
        for dim,groups in (('RX',5),('day',2)):
            for index in range(groups):
                part=np.eye(6,dtype=np.int64)*(n//6//groups)
                mistakes=errors//groups+int(index<errors%groups);part[0,0]-=mistakes;part[0,1]+=mistakes
                records.append(record(part,view=view,dimension=dim,value=str(index)))
    return dict(status='ANALYZED',role=role,count=n,F_seed=2026092701,original_default_head=True,
                F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',target_access=role=='target',
                all_predictions_before_truth=True,training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,
                records=records,views={v:dict(restored=dict(waveform=dict(iq_nmse=.02,prefix232_nmse=.02,tail24_nmse=.02,fft_log_power_L1=.1))) for v in d.VIEWS})


def test_recomputes_every_partition_and_detects_one_target_error_regression():
    baseline=summary_metrics(summary('target',errors=100),'target')
    candidate=summary_metrics(summary('target',errors=101),'target')
    result=compare(candidate,baseline)
    assert not result['accuracy_nonregression']
    assert all(v['correct_delta']==-1 and v['accuracy_delta_pp']==-100/168000 for v in result['views'].values())
    assert baseline['independently_recomputed_records']==7*(1+6+5+2)
    assert compare(baseline,baseline)['accuracy_nonregression']
    assert sum(sum(row) for row in baseline['views']['clean']['confusion'])==168000


@pytest.mark.parametrize('key,bad',[('accuracy',99.),('macro_f1',99.),('net_repaired',0),('count',1)])
def test_inconsistent_aggregate_arithmetic_rejected(key,bad):
    value=summary()['records'][0];value[key]=bad
    with pytest.raises(ValueError):classification_record(value)


def test_incomplete_or_mismatched_physical_partitions_rejected():
    value=summary();value['records']=[r for r in value['records'] if not(r['view']=='clean' and r['dimension']=='RX' and r['value']=='0')]
    with pytest.raises(ValueError):summary_metrics(value,'source')
    with pytest.raises(ValueError):summary_metrics(summary('target'),'source')


def test_preserves_paired_clean_feature_diagnostics_and_grouped_NMSE():
    value=summary()
    for view in d.VIEWS:
        value['views'][view]['restored'].update(paired_clean_feature_cosine_distance=.03,
            paired_clean_vs_other_TX={'aggregate_margin':.2})
    value['records'][0]['iq_nmse']=.02
    result=summary_metrics(value,'source')
    assert result['views']['clean']['paired_clean_geometry']['paired_clean_feature_cosine_distance']==.03
    assert result['views']['clean']['paired_clean_geometry']['paired_clean_vs_other_TX']=={'aggregate_margin':.2}
    assert result['views']['clean']['iq_nmse']==.02


def training(arm):
    rows=[dict(epoch=e,updates=247*e,L_exposures=6300*e,U_exposures=56700*e,teacher_IQ_examples=63000*e,
               seconds=100.*e,clip_fraction=0.,losses=dict(loss=.31,iq_nmse=.1,teacher_stage_nmse=.001,task_KL=.1,task_feature_distance=.2)) for e in range(1,21)]
    cal=512 if arm=='activation_task' else 0
    complete=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        teacher_IQ_examples=1260000,inherited_prior_epochs=20,cumulative_prior_epochs=40,cumulative_serial_training_passes=80,
        parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm]-2157547,
        calibration_teacher_IQ_examples=cal,total_teacher_IQ_examples=1260000+cal,calibration_L_exposures=55 if cal else 0,
        calibration_U_exposures=457 if cal else 0,calibration_seconds=1.,teacher_parameter_updates=0,frozen_context_updates=0,
        TX_labels_read=0,F_outputs_read=2520000,F_weights_read=1,training_F_access=True,F_teacher_forward_examples=1260000,
        F_student_forward_examples=1260000,F_parameter_updates=0,head_fit_updates=0,V_reads=0,target_reads=0)
    return rows,complete


@pytest.mark.parametrize('arm',d.ARMS)
def test_full_training_cost_and_frozen_F_knowledge_are_honest(arm):
    rows,complete=training(arm)
    result=training_metrics(rows,complete,arm)
    assert result['full_log_records']==20 and set(result['losses'])==set(rows[0]['losses'])
    assert result['training_complete']['training_F_access']
    for key,bad in [('F_outputs_read',0),('F_weights_read',0),('training_F_access',False),('F_parameter_updates',1),('head_fit_updates',1),('total_teacher_IQ_examples',1260511)]:
        with pytest.raises(ValueError):training_metrics(rows,dict(complete,**{key:bad}),arm)
    invalid=copy.deepcopy(rows);invalid[-1]['losses']['loss']+=.1
    with pytest.raises(ValueError,match='objective'):training_metrics(invalid,complete,arm)
    with pytest.raises(ValueError):training_metrics(rows[:-1],complete,arm)


def task_summary(role='source',errors=180):
    return dict(summary(role,errors),training_F_access=True)


def test_task_summary_cannot_claim_F_free_training():
    value=task_summary()
    with pytest.raises(ValueError):summary_metrics(value,'source')
    result=summary_metrics(value,'source',training_F_access=True)
    assert result['views']['clean']['correct']==1620
    with pytest.raises(ValueError):summary_metrics(summary(),'source',training_F_access=True)


@pytest.mark.parametrize('source_qualified',[True,False])
def test_fixed_source_selection_retains_legacy_zero_drop_and_new_user_target_allowance(source_qualified):
    baseline=summary();candidate=task_summary(errors=198 if source_qualified else 199)
    decision=select_source({a:candidate for a in d.ARMS},baseline)
    from experiments.pamf_r17_calibrated.calibration import smoke_info
    from experiments.pamf_r17_task.calibration import none_info
    from experiments.pamf_r17_calibrated import design as cal_d
    arms={}
    for arm in d.ARMS:
        rows,complete=training(arm)
        profile=dict(status='PROFILE_COMPLETE',includes_full_frozen_context_and_CG=True,source_V_IQ=16,TX_labels_read=0,state_updates=0,
            results={key:dict(samples=16,warmup=2,parameters=p,median_ms=10.,p90_ms=12.,peak_allocated_bytes=1000)
                     for key,p in (('original_R17',4665073),('compact',d.PARAMETERS[arm]))})
        if arm=='activation_task':
            calibration=dict(smoke_info(),mode='source_train512',examples=512,observed_examples=512,L_s=55,U_s=457,received_epoch=1,
                selection_seed=cal_d.CALIBRATION_SEED,eligible_layers=84,sampled_patches_min=4096,sampled_patches_max=4096,seconds=1.,
                factor_reports=[dict(regularized_output_relative_squared_error=.1,factor_kernel_parameters=10,original_kernel_parameters=20) for _ in range(84)])
        else:calibration=dict(none_info(),observed_examples=0,eligible_layers=0,sampled_patches_min=0,sampled_patches_max=0,factor_reports=[],seconds=1.)
        arms[arm]=dict(calibration=calibration,source_summary=candidate,training_records=rows,training_complete=complete,
                       matched_profile=profile,checkpoint_bytes=100,target_summary=None,assessment=None)
    if source_qualified:arms['step_norm_task'].update(target_summary=task_summary('target',101),assessment={'target_accuracy_nonregression':False})
    collection=dict(run_id=d.RUN,method_commit='a310b3b5ff30e23584e7bc950e8a0ff96ac685fb',failure=None,
        complete=dict(status='ANALYZED',target_rows=decision['target_rows']),actual_live_owned_PIDs=[],arms=arms,
        baseline_source=baseline,baseline_target=summary('target',100) if source_qualified else None,source_decision=decision)
    result=completed_metrics(collection)
    assert result['target_performance_evaluated']==source_qualified and not result['goal_complete']
    assert result['target_accuracy_nonregression'] is (False if source_qualified else None)
    assert result['source_screening_allowance_pp']==1 and result['target_acceptance_allowance_pp']==1
    assert result['launch_time_target_acceptance_allowance_pp']==0
    assert result['target_user_accepted'] is (True if source_qualified else None)
    if source_qualified:
        assert decision['checks']['step_norm_task']['qualified']
        assert not all(decision['checks']['step_norm_task']['per_view_accuracy_nonregression'].values())
        assert result['arms']['step_norm_task']['target_vs_R17']['views']['clean']['correct_delta']==-1
