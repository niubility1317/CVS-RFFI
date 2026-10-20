"""Recompute fixed two-row target comparison from completed aggregates only."""
from experiments.pamf_r17_target_compare import design as d
from tools.analyze_pamf_r17_task_results import summary_metrics,compare,profile_metrics
from tools.r17_compact_acceptance import target_acceptance

METHOD='81c39cb908373d4e19b70ec8406f4083529da99a'


def completed_metrics(value):
    if (value.get('run_id')!=d.RUN or value.get('method_commit')!=METHOD or value.get('failure') is not None
            or value.get('actual_live_owned_PIDs') or value.get('comparison_plan')!=d.configuration()
            or value.get('complete')!={'status':'ANALYZED','target_rows':list(d.ARMS),'new_training_updates':0,'target_feedback':False}
            or set(value.get('rows',{}))!=set(d.ARMS) or value.get('smoke',{}).get('status')!='PASS'):
        raise ValueError('Both fixed completed rows and exited actual handles required')
    baseline=summary_metrics(value['baseline_target'],'target')
    public=dict(run_id=d.RUN,method_commit=METHOD,status='ANALYZED',source_run=d.SOURCE_RUN,source_commit=d.SOURCE_COMMIT,
        baseline_target=baseline,target_acceptance_allowance_pp=1.,new_training_updates=0,head_fit_updates=0,
        original_source_waveform_qualified=False,target_feedback=False,goal_complete=False,rows={},accepted_rows=[],
        limitation='Two frozen models on one exposed fixed target benchmark/originalF50seed2701;notnewblindtest/globalminimum/multiseed or inorbit guarantee')
    for arm in d.ARMS:
        row=value['rows'][arm];done=row['prediction_complete']
        expected=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role='target',count=168000,views=list(d.VIEWS),
            readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.source.F_RUN,F_seed=d.source.F_SEED)
        if any(done.get(k)!=v for k,v in expected.items()) or row['complete']!=dict(status='ANALYZED',arm=arm,count=168000,all_predictions_before_truth=True,target_feedback=False):
            raise ValueError('Complete seven-view independent truth-last target row required')
        target=summary_metrics(row['summary'],'target',training_F_access=True)
        delta=compare(target,baseline);acceptance=target_acceptance(target,baseline);assessment=row['assessment']
        if (assessment.get('status')!='ANALYZED' or assessment.get('arm')!=arm or assessment.get('user_acceptance')!=acceptance
                or assessment.get('target_vs_R17')!=delta or assessment.get('original_source_waveform_qualified') is not False
                or any(assessment.get(k)!=0 for k in ('training_updates','head_fit_updates')) or assessment.get('target_feedback') is not False):
            raise ValueError('Actual independent assessment differs from recomputed exact counts')
        public['rows'][arm]=dict(parameters=d.PARAMETERS[arm],parameter_reduction_percent=100*(1-d.PARAMETERS[arm]/4665073),
            checkpoint_bytes=row['checkpoint_bytes'],target=target,target_vs_R17=delta,user_acceptance=acceptance,
            prediction_seconds=done['seconds'],per_view_prediction_seconds=done['per_view_seconds'],
            inherited_source_profile=profile_metrics(row['source_profile'],arm))
        if acceptance['accepted']:public['accepted_rows'].append(arm)
    return public
