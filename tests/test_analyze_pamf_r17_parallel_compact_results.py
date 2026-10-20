import runpy
from pathlib import Path
import pytest
from experiments.pamf_r17_parallel_compact import design as d
from tools.analyze_pamf_r17_parallel_compact_results import completed_metrics,METHOD
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

old=runpy.run_path(str(Path(__file__).with_name('test_analyze_pamf_r17_task_results.py')))


def fixture():
    original=old['summary']('target',100);baseline=summary_metrics(original,'target');rows={}
    for arm,errors in zip(d.ARMS,(1780,1781,1780,1781)):
        records,trained=old['training']('step_norm_task')
        for key in tuple(trained):
            if key.startswith('calibration_'):trained.pop(key)
        trained.update(arm=arm,parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm]-2157547,initialization_IQ_examples=0,initialization_seconds=1.)
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
