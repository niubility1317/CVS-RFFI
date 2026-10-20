"""Recalculate completed R46 aggregates; no IQ, labels, GPU, or model selection."""
import math
import numpy as np
from experiments.pamf_r44_pruned_refine import design as d
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

METHOD='bbb8daf817de050b958d217b70d4c33f4aefe341'


def training_metrics(records,complete,arm):
    if arm not in d.ARMS or len(records)!=20:raise ValueError('All twenty fixed training records required')
    previous=0.
    for epoch,row in enumerate(records,1):
        expected=dict(epoch=epoch,updates=247*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,
            teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,teacher_weight=d.teacher_weight(epoch))
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Actual source exposure or teacher schedule differs')
        loss=row['losses']
        if set(loss)!={'loss','iq_nmse','spectral_error','teacher_stage_nmse'} or any(not math.isfinite(v) or v<0 for v in loss.values()):
            raise ValueError('Only finite registered F-free losses allowed')
        if not math.isclose(loss['loss'],loss['iq_nmse']+.05*loss['spectral_error']+d.teacher_weight(epoch)*loss['teacher_stage_nmse'],rel_tol=1e-5,abs_tol=1e-7):
            raise ValueError('Weighted objective does not close')
        if not math.isfinite(row['seconds']) or row['seconds']<=previous or not 0<=row['clip_fraction']<=1:
            raise ValueError('Invalid monotone training time or clipping')
        if not math.isclose(row['learning_rate_last'],d.learning_rate(247*epoch-1),rel_tol=1e-9):raise ValueError('Learning-rate differs')
        previous=row['seconds']
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,
        parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm],context_parameters=d.CONTEXT_PARAMETERS[arm],
        context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,context_trainable=True,
        cumulative_coarse_epochs=80,cumulative_action_epochs=80,cumulative_prior_epochs=80,
        cumulative_component_epoch_sum=240,cumulative_serial_training_passes=120,initialization_IQ_examples=0,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True)
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Complete fixed budget or training permissions differ')
    losses={}
    for key in ('loss','iq_nmse','spectral_error','teacher_stage_nmse'):
        values=np.asarray([r['losses'][key] for r in records],dtype=np.float64)
        losses[key]=dict(first=float(values[0]),last=float(values[-1]),minimum=float(values.min()),maximum=float(values.max()),
            mean=float(values.mean()),slope_per_epoch=float(np.polyfit(np.arange(1,21),values,1)[0]))
    return dict(full_log_records=20,records=records,losses=losses,training_complete=complete,
        epoch_seconds_median=float(np.median(np.diff([0.,*[r['seconds'] for r in records]]))),checkpoint_selection='fixed final E20/4940')


def profile_metrics(profile,arm):
    expected=dict(status='PROFILE_COMPLETE',includes_full_context_and_CG=True,checkpoint_payload_on_GPU=False,
        optimizer_on_GPU=False,source_V_IQ=16,TX_labels_read=0,state_updates=0,physical_GPU=d.GPUS[arm])
    if any(profile.get(k)!=v for k,v in expected.items()):raise ValueError('Matched same-GPU full-chain profile required')
    sizes=dict(original_R17=4665073,parent_R44_442805=442805,compact=d.PARAMETERS[arm])
    if set(profile['results'])!=set(sizes):raise ValueError('All three profiles required')
    for name,size in sizes.items():
        item=profile['results'][name]
        if (item['samples'],item['warmup'],item['parameters'])!=(16,2,size):raise ValueError('Profile scope or count differs')
        if any(not math.isfinite(item[k]) or item[k]<=0 for k in ('median_ms','p90_ms','resident_before_forward_bytes','peak_allocated_bytes')):
            raise ValueError('Finite actual cost required')
        if item['incremental_peak_bytes']!=item['peak_allocated_bytes']-item['resident_before_forward_bytes']:raise ValueError('Peak delta differs')
    return dict(**profile,latency_ratio_to_R44=profile['results']['compact']['median_ms']/profile['results']['parent_R44_442805']['median_ms'],
        limitation='16 source V observations,batch1;sequential ordering and concurrent load affect latency')


def completed_metrics(value):
    if (value.get('run_id')!=d.RUN or value.get('method_commit')!=METHOD or value.get('failure') is not None
            or value.get('actual_live_owned_PIDs') or value.get('complete',{}).get('status')!='ANALYZED'
            or value['complete'].get('source_rows')!=list(d.ARMS) or value['complete'].get('target_rows')!=list(d.ARMS)
            or set(value.get('rows',{}))!=set(d.ARMS)):
        raise ValueError('Both actual fixed rows completed and owned handles exited required')
    bases={name:{role:summary_metrics(raw[role],role) for role in ('source','target')} for name,raw in value['baselines'].items()}
    if set(bases)!={'R17','R38','R44'}:raise ValueError('Both original references required')
    result=dict(status='ANALYZED',run_id=d.RUN,method_commit=METHOD,baselines=bases,rows={},
        target_feedback=False,goal_complete=False,limitation='One model seed;human-designated exposed engineering baseline;not independent blind validation')
    for arm,row in value['rows'].items():
        if row.get('complete',{}).get('status')!='ANALYZED' or row.get('failure') is not None:raise ValueError('Complete same-row scoring required')
        metrics={}
        for role in ('source','target'):
            expected=dict(status='PREDICTIONS_COMPLETE',role=role,count=1800 if role=='source' else 168000,
                views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED)
            if any(row[role+'_prediction_complete'].get(k)!=v for k,v in expected.items()):raise ValueError('Same-head full-seven predictions required')
            metrics[role]=summary_metrics(row[role+'_summary'],role)
        acceptance={name:target_acceptance(metrics['target'],base['target']) for name,base in bases.items()}
        acceptance['R38']['reference']='R38_794529_report_only'
        acceptance['R44']['reference']='human_R44_442805'
        deltas={name:compare(metrics['target'],base['target']) for name,base in bases.items()}
        gains={view:deltas['R44']['views'][view]['accuracy_delta_pp'] for view in d.VIEWS}
        gap={view:84.67619047619047-metrics['target']['views'][view]['accuracy'] for view in d.VIEWS if view!='clean'}
        result['rows'][arm]=dict(parameters=d.PARAMETERS[arm],context_parameters=d.CONTEXT_PARAMETERS[arm],
            reduction_from_R38_percent=100*(1-d.PARAMETERS[arm]/794529),reduction_from_R44_percent=100*(1-d.PARAMETERS[arm]/442805),training_F_access=False,
            training=training_metrics(row['training_records'],row['training_complete'],arm),source=metrics['source'],target=metrics['target'],
            target_vs_R17=deltas['R17'],target_vs_R44=deltas['R44'],target_vs_R38=deltas['R38'],acceptance=acceptance,
            accepted_both=acceptance['R17']['accepted'] and acceptance['R44']['accepted'],profile=profile_metrics(row['matched_profile'],arm),
            checkpoint_bytes=row['checkpoint_bytes'],stdout_findings=row.get('stdout_findings',[]),
            performance_optimization=dict(improved_views=[v for v,g in gains.items() if g>0],decreased_views=[v for v,g in gains.items() if g<0],
                low_urban_accuracy=metrics['target']['views']['practical_low_urban']['accuracy'],
                low_urban_at_least_70=metrics['target']['views']['practical_low_urban']['accuracy']>=70,
                other_five_within_5pp_of_raw_clean=all(g<=5 for v,g in gap.items() if v!='practical_low_urban'),practical_gap_to_raw_clean_pp=gap,
                acceptance_is_not_optimization_success=True))
    return result
