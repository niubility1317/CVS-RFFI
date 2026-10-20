"""Analyze the fixed R09 joint optimization matrix after all four rows complete, using aggregates only."""
import math
import numpy as np
from experiments.pamf_r17_joint_optimize import design as d
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

METHOD='63a8b45b502351ba72aaf80f83b377036ce255fe'


def training_metrics(records,complete,arm):
    if arm not in d.ARMS or len(records)!=20:raise ValueError('All twenty fixed training records required')
    names=('loss','iq_nmse','teacher_stage_nmse','task_KL','task_feature_distance');last=0.
    for epoch,row in enumerate(records,1):
        if (row['epoch'],row['updates'],row['L_exposures'],row['U_exposures'],row['teacher_IQ_examples'])!=(epoch,247*epoch,6300*epoch,56700*epoch,63000*epoch):
            raise ValueError('Complete source exposure required')
        if not math.isfinite(row['seconds']) or row['seconds']<=last or not 0<=row['clip_fraction']<=1:raise ValueError('Finite training progress required')
        last=row['seconds'];loss=row['losses']
        if set(loss)!=set(names) or any(not math.isfinite(v) or v<0 for v in loss.values()):raise ValueError('All registered losses required')
        if not math.isclose(loss['loss'],10*loss['teacher_stage_nmse']+loss['task_KL']+loss['task_feature_distance']+float(d.CLEAN_OPT[arm])*loss['iq_nmse'],rel_tol=1e-5,abs_tol=1e-7):
            raise ValueError('Actual objective differs from fixed distillation')
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        teacher_IQ_examples=1260000,inherited_prior_epochs=20,cumulative_prior_epochs=40,cumulative_serial_training_passes=80,
        initialization_IQ_examples=0,total_teacher_IQ_examples=1260000,context_parameters=d.CONTEXT_PARAMETERS,
        teacher_context_forward_examples=1260000,student_context_forward_examples=1260000,parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm],
        teacher_parameter_updates=0,context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,context_trainable=True,
        inherited_coarse_epochs=20,inherited_action_epochs=20,cumulative_coarse_epochs=40,cumulative_action_epochs=40,cumulative_component_epoch_sum=120,
        clean_IQ_loss_weight=float(d.CLEAN_OPT[arm]),F_training_reference='source_paired_clean' if d.CLEAN_OPT[arm] else 'original_R17_restored',TX_labels_read=0,F_outputs_read=2520000,F_weights_read=1,
        training_F_access=True,F_teacher_forward_examples=1260000,F_student_forward_examples=1260000,F_parameter_updates=0,head_fit_updates=0,V_reads=0,target_reads=0)
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Final full source/F knowledge budget and permissions required')
    if not math.isfinite(complete.get('initialization_seconds',-1)) or complete.get('initialization_seconds',-1)<0:
        raise ValueError('Actual data-free initialization time required')
    losses={}
    for name in names:
        values=np.asarray([r['losses'][name] for r in records],np.float64)
        losses[name]=dict(first=float(values[0]),last=float(values[-1]),minimum=float(values.min()),maximum=float(values.max()),
            mean=float(values.mean()),slope_per_epoch=float(np.polyfit(np.arange(1,21),values,1)[0]))
    intervals=np.diff([0.,*[r['seconds'] for r in records]])
    return dict(full_log_records=20,losses=losses,epoch_seconds_median=float(np.median(intervals)),training_complete=complete,
        checkpoint_selection='fixedfinalE20,nottraining/V/targetbest')


def profile_metrics(value,arm):
    expected=dict(status='PROFILE_COMPLETE',includes_full_frozen_context_and_CG=True,checkpoint_payload_on_GPU=False,
        optimizer_on_GPU=False,source_V_IQ=16,TX_labels_read=0,state_updates=0)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Payload-free same-source profile required')
    results=value['results']
    if set(results)!= {'original_R17','compact'}:raise ValueError('Both profile models required')
    for key,parameters in (('original_R17',4665073),('compact',d.PARAMETERS[arm])):
        row=results[key]
        if row['samples']!=16 or row['warmup']!=2 or row['parameters']!=parameters:raise ValueError('Matched full inference profile required')
        if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('median_ms','p90_ms','resident_before_forward_bytes','peak_allocated_bytes')):
            raise ValueError('Finite inference time and resident memory required')
        if row['incremental_peak_bytes']!=row['peak_allocated_bytes']-row['resident_before_forward_bytes'] or row['incremental_peak_bytes']<0:
            raise ValueError('Memory baseline/increment mismatch')
    return dict(**value,median_latency_ratio=results['compact']['median_ms']/results['original_R17']['median_ms'],
        peak_allocated_ratio=results['compact']['peak_allocated_bytes']/results['original_R17']['peak_allocated_bytes'],
        scope='checkpointpayload/optimizerexcluded;includesmodel/input/context/CG;source-only/coowner/ordercaveats')


def completed_metrics(value):
    expected_done=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),no_source_score_selection=True,target_feedback=False,goal_complete=False)
    if (value.get('run_id')!=d.RUN or value.get('method_commit')!=METHOD or value.get('failure') is not None
            or value.get('actual_live_owned_PIDs') or value.get('complete')!=expected_done or value.get('comparison_plan')!=d.configuration()
            or set(value.get('rows',{}))!=set(d.ARMS) or value.get('smoke',{}).get('status')!='PASS'):
        raise ValueError('All four fixed rows complete and actual handles exited required')
    baseline={role:summary_metrics(value['baseline_'+role],role) for role in ('source','target')}
    public=dict(run_id=d.RUN,method_commit=METHOD,status='ANALYZED',baseline_source=baseline['source'],baseline_target=baseline['target'],
        target_acceptance_allowance_pp=1.,training_F_access=True,head_fit_updates=0,target_feedback=False,goal_complete=False,
        rows={},accepted_rows=[],limitation='Singletrainingseed/originalF50seed2701/exposedbenchmark;notglobalminimum/newblindtest/Phase2/inorbit')
    for arm in d.ARMS:
        row=value['rows'][arm];metrics={}
        for role in ('source','target'):
            done=row[role+'_prediction_complete'];expected=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=1800 if role=='source' else 168000,
                views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED)
            if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete seven-view legal predictions required')
            metrics[role]=summary_metrics(row[role+'_summary'],role,training_F_access=True)
        acceptance=target_acceptance(metrics['target'],baseline['target']);delta=compare(metrics['target'],baseline['target'])
        assessment=row['assessment'];expected=dict(status='ANALYZED',arm=arm,user_acceptance=acceptance,target_vs_R17=delta,
            full_parameters=d.PARAMETERS[arm],original_parameters=4665073,target_feedback=False)
        if any(assessment.get(k)!=v for k,v in expected.items()):raise ValueError('Independent assessment and exact counts disagree')
        expected_done=dict(status='ANALYZED',arm=arm,role='target',all_predictions_before_truth=True,target_feedback=False)
        if row['complete']!=expected_done:raise ValueError('Complete independent target score required')
        public['rows'][arm]=dict(parameters=d.PARAMETERS[arm],parameter_reduction_percent=100*(1-d.PARAMETERS[arm]/4665073),
            checkpoint_bytes=row['checkpoint_bytes'],initialization=row['initialization'],
            training=training_metrics(row['training_records'],row['training_complete'],arm),
            source=metrics['source'],source_vs_R17=compare(metrics['source'],baseline['source']),
            target=metrics['target'],target_vs_R17=delta,user_acceptance=acceptance,profile=profile_metrics(row['matched_profile'],arm),
            target_prediction_seconds=row['target_prediction_complete']['seconds'])
        gains={view:100*(metrics['target']['views'][view]['correct']-baseline['target']['views'][view]['correct'])/168000 for view in d.VIEWS}
        low=metrics['target']['views']['practical_low_urban']['accuracy']
        public['rows'][arm]['performance_optimization']=dict(accuracy_delta_pp=gains,
            improved_views=[view for view in d.VIEWS if gains[view]>0],unchanged_views=[view for view in d.VIEWS if gains[view]==0],
            decreased_views=[view for view in d.VIEWS if gains[view]<0],all_seven_non_decreasing=all(v>=0 for v in gains.values()),
            low_urban_accuracy=low,low_urban_at_least_70=low>=70,
            practical_gap_to_same_model_clean_pp={view:metrics['target']['views']['clean']['accuracy']-metrics['target']['views'][view]['accuracy'] for view in d.VIEWS if view!='clean'},
            acceptance_is_not_optimization_success=True,near_clean_threshold_unspecified=True,single_seed_only=True)
        if acceptance['accepted']:public['accepted_rows'].append(arm)
    return public
