import runpy
from pathlib import Path
import pytest
from experiments.pamf_r17_joint_optimize import design as d
from tools.analyze_pamf_r17_joint_optimize_results import completed_metrics,METHOD
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

old=runpy.run_path(str(Path(__file__).with_name('test_analyze_pamf_r17_task_results.py')))


def fixture():
    original=old['summary']('target',100);baseline=summary_metrics(original,'target');rows={}
    for arm,errors in zip(d.ARMS,(1780,1781,1780,1781)):
        records,trained=old['training']('step_norm_task')
        for key in tuple(trained):
            if key.startswith('calibration_'):trained.pop(key)
        trained.update(arm=arm,parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm],initialization_IQ_examples=0,initialization_seconds=1.,context_parameters=d.CONTEXT_PARAMETERS,teacher_context_forward_examples=1260000,student_context_forward_examples=1260000)
        trained.pop('frozen_context_updates',None)
        trained.update(context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,context_trainable=True,
            inherited_coarse_epochs=20,inherited_action_epochs=20,cumulative_coarse_epochs=40,cumulative_action_epochs=40,cumulative_component_epoch_sum=120,
            clean_IQ_loss_weight=float(d.CLEAN_OPT[arm]),F_training_reference='source_paired_clean' if d.CLEAN_OPT[arm] else 'original_R17_restored')
        for record in records:record['losses']['loss']+=float(d.CLEAN_OPT[arm])*record['losses']['iq_nmse']
        target=dict(old['summary']('target',errors),training_F_access=True);metrics=summary_metrics(target,'target',True)
        profile=dict(status='PROFILE_COMPLETE',includes_full_frozen_context_and_CG=True,checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,
            source_V_IQ=16,TX_labels_read=0,state_updates=0,results={key:dict(parameters=p,samples=16,warmup=2,median_ms=10.,p90_ms=12.,
                resident_before_forward_bytes=500,peak_allocated_bytes=1000,incremental_peak_bytes=500) for key,p in (('original_R17',4665073),('compact',d.PARAMETERS[arm]))})
        row=dict(training_records=records,training_complete=trained,initialization={'IQ_examples':0},checkpoint_bytes=100,matched_profile=profile,
            source_summary=dict(old['summary'](),training_F_access=True),target_summary=target,
            complete=dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False),
            assessment=dict(status='ANALYZED',arm=arm,user_acceptance=target_acceptance(metrics,baseline),target_vs_R17=compare(metrics,baseline),
                full_parameters=d.PARAMETERS[arm],original_parameters=4665073,target_feedback=False))
        for role in ('source','target'):
            row[role+'_prediction_complete']=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=1800 if role=='source' else 168000,
                views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED,seconds=100.)
        rows[arm]=row
    return dict(run_id=d.RUN,method_commit=METHOD,failure=None,actual_live_owned_PIDs=[],comparison_plan=d.configuration(),
        complete=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),no_source_score_selection=True,target_feedback=False,goal_complete=False),
        smoke=dict(status='PASS'),baseline_source=old['summary'](),baseline_target=original,rows=rows)


def test_fixed_rows_source_diagnostic_only_and_exact_one_extra_error_rejects():
    result=completed_metrics(fixture())
    assert result['accepted_rows']==[d.ARMS[0],d.ARMS[2]] and not result['goal_complete']
    assert all(row['training']['full_log_records']==20 for row in result['rows'].values())
    assert result['rows'][d.ARMS[0]]['source']['views']['clean']['accuracy']<100


@pytest.mark.parametrize('bad',['missingrow','live','missingview','assessment','trainingbudget','trainingloss','profilepayload','profilebaseline'])
def test_incomplete_or_inconsistent_evidence_cannot_be_reported_complete(bad):
    value=fixture();row=value['rows'][d.ARMS[0]]
    if bad=='missingrow':value['rows'].pop(d.ARMS[-1])
    elif bad=='live':value['actual_live_owned_PIDs']=[123]
    elif bad=='missingview':row['target_prediction_complete']['views'].pop()
    elif bad=='assessment':row['assessment']['user_acceptance']['accepted']=False
    elif bad=='trainingbudget':row['training_complete']['updates']-=1
    elif bad=='trainingloss':row['training_records'][-1]['losses']['loss']+=1
    elif bad=='profilepayload':row['matched_profile']['optimizer_on_GPU']=True
    else:row['matched_profile']['results']['compact']['incremental_peak_bytes']=499
    with pytest.raises(ValueError):completed_metrics(value)


@pytest.mark.parametrize('bad',['frozencontext','partialoptimizer','wrongFreference','wrongcleanIQweight','componentepochs','coarseupdates'])
def test_joint_objective_and_full_component_budget_are_not_prior_only(bad):
    value=fixture();row=value['rows']['joint_clean_quarter'];trained=row['training_complete']
    if bad=='frozencontext':trained['context_trainable']=False
    elif bad=='partialoptimizer':trained['trainable_parameters']-=d.CONTEXT_PARAMETERS
    elif bad=='wrongFreference':trained['F_training_reference']='original_R17_restored'
    elif bad=='wrongcleanIQweight':trained['clean_IQ_loss_weight']=0.
    elif bad=='componentepochs':trained['cumulative_component_epoch_sum']=80
    else:trained['coarse_updates']=0
    with pytest.raises(ValueError):completed_metrics(value)


def test_accepted_compactness_does_not_get_reported_as_optimization_success():
    result=completed_metrics(fixture());row=result['rows'][d.ARMS[0]]
    outcome=row['performance_optimization']
    assert row['user_acceptance']['accepted'] and outcome['acceptance_is_not_optimization_success']
    assert not outcome['improved_views'] and outcome['decreased_views']==list(d.VIEWS)
    assert not outcome['all_seven_non_decreasing'] and outcome['near_clean_threshold_unspecified']
    assert outcome['low_urban_at_least_70']==(outcome['low_urban_accuracy']>=70)


def test_same_architecture_objective_comparison_keeps_direction_and_all_views():
    result=completed_metrics(fixture())
    assert set(result['matched_objective_comparisons'])=={'quarter','eighth_pointwise'}
    for pair in result['matched_objective_comparisons'].values():
        assert pair['delta_direction']=='clean_corrective_minus_teacher_fidelity'
        assert set(pair['target']['views'])==set(d.VIEWS)
        assert set(pair['source']['views'])==set(d.VIEWS)
        for item in pair['target']['views'].values():
            assert item['correct_delta']==-1
            assert item['accuracy_delta_pp']==-100/168000
        assert pair['parameters']==d.PARAMETERS[pair['teacher_arm']]==d.PARAMETERS[pair['clean_arm']]
        assert pair['changed_factors']==['paired_source_clean_IQ_loss','F_reference_from_R17_restored_to_source_clean']
        assert not pair['single_factor_attribution'] and not pair['target_feedback']
        assert pair['raw_sample_pairwise_correctness_not_inferred_from_aggregate']
