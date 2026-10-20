import copy
import runpy
from pathlib import Path
import pytest
from experiments.pamf_r17_target_compare import design as d
from tools.analyze_pamf_r17_target_comparison_results import completed_metrics,METHOD
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

summary=runpy.run_path(str(Path(__file__).with_name('test_analyze_pamf_r17_task_results.py')))['summary']


def fixture():
    baseline=summary('target',100);original=summary_metrics(baseline,'target');rows={}
    for arm,errors in zip(d.ARMS,(1780,1781)):
        current=dict(summary('target',errors),training_F_access=True);metrics=summary_metrics(current,'target',True)
        profile=dict(status='PROFILE_COMPLETE',includes_full_frozen_context_and_CG=True,source_V_IQ=16,TX_labels_read=0,state_updates=0,
            results={key:dict(samples=16,warmup=2,parameters=p,median_ms=10.,p90_ms=12.,peak_allocated_bytes=1000)
                     for key,p in (('original_R17',4665073),('compact',d.PARAMETERS[arm]))})
        rows[arm]=dict(summary=current,checkpoint_bytes=100,source_profile=profile,
            prediction_complete=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role='target',count=168000,views=list(d.VIEWS),readouts=['restored'],
                TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.source.F_RUN,F_seed=d.source.F_SEED,seconds=100.,per_view_seconds={v:10. for v in d.VIEWS}),
            complete=dict(status='ANALYZED',arm=arm,count=168000,all_predictions_before_truth=True,target_feedback=False),
            assessment=dict(status='ANALYZED',arm=arm,user_acceptance=target_acceptance(metrics,original),target_vs_R17=compare(metrics,original),
                original_source_waveform_qualified=False,training_updates=0,head_fit_updates=0,target_feedback=False))
    return dict(run_id=d.RUN,method_commit=METHOD,failure=None,actual_live_owned_PIDs=[],comparison_plan=d.configuration(),
        complete=dict(status='ANALYZED',target_rows=list(d.ARMS),new_training_updates=0,target_feedback=False),smoke=dict(status='PASS'),
        baseline_target=baseline,rows=rows)


def test_all_fixed_rows_are_reported_and_one_excess_error_is_not_rounded_away():
    result=completed_metrics(fixture())
    assert set(result['rows'])==set(d.ARMS) and result['accepted_rows']==['step_norm_task']
    assert result['rows']['step_norm_task']['user_acceptance']['accepted']
    assert not result['rows']['activation_task']['user_acceptance']['accepted']
    assert not result['rows']['step_norm_task']['target_vs_R17']['accuracy_nonregression']
    assert not result['goal_complete'] and result['new_training_updates']==0


@pytest.mark.parametrize('bad',['missing_row','live_worker','missing_view','changed_assessment'])
def test_incomplete_or_inconsistent_evidence_is_rejected(bad):
    value=fixture()
    if bad=='missing_row':value['rows'].pop(d.ARMS[-1])
    elif bad=='live_worker':value['actual_live_owned_PIDs']=[123]
    elif bad=='missing_view':value['rows'][d.ARMS[0]]['prediction_complete']['views'].pop()
    else:value['rows'][d.ARMS[-1]]['assessment']['user_acceptance']['accepted']=True
    with pytest.raises(ValueError):completed_metrics(value)
