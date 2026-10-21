"""Validate one shared source trajectory and all three frozen formal readouts."""
import math
import numpy as np
from experiments.pamf_r46_weight_mean import design as d
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

METHOD='21141083f91ca7703b9242e5d90873666a8414b6'


def training_metrics(records,complete):
    if len(records)!=20:raise ValueError('All twenty shared source epochs required')
    previous=host=0.
    for epoch,row in enumerate(records,1):
        expected=dict(epoch=epoch,updates=247*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,
            teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,teacher_weight=d.teacher_weight(epoch),
            microbatch=256,source_tensor_backend='direct_float32_buffer',EMA_updates=max(0,epoch-10)*247,
            late_mean_snapshots=max(0,epoch-15),shared_source_training_paths=1)
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Shared source exposure or averaging schedule differs')
        terms=row['losses'];keys={'loss','iq_nmse','spectral_error','teacher_stage_nmse'}
        if set(terms)!=keys or any(not math.isfinite(v) or v<0 for v in terms.values()):raise ValueError('Finite original F-free losses required')
        if not math.isclose(terms['loss'],terms['iq_nmse']+.05*terms['spectral_error']+d.teacher_weight(epoch)*terms['teacher_stage_nmse'],rel_tol=1e-5,abs_tol=1e-7):raise ValueError('Original objective must close')
        if not math.isclose(row['learning_rate_last'],d.learning_rate(247*epoch-1),rel_tol=1e-9):raise ValueError('Original global LR required')
        seconds=row['seconds'];enqueue=row['averaging_host_enqueue_seconds']
        if not math.isfinite(seconds) or seconds<=previous or not math.isfinite(enqueue) or enqueue<host or not 0<=row['clip_fraction']<=1:raise ValueError('Actual monotone cost required')
        previous,host=seconds,enqueue
    expected=dict(status='SOURCE_TRAINED',arm=d.SOURCE_ARM,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,shared_source_training_paths=1,
        training_counted_once_across_readouts=True,readouts=list(d.ARMS),parameters=273765,trainable_parameters=273765,
        context_parameters=185615,context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,
        context_trainable=True,paired_clean_optimized=True,prior_parameters_changed=True,EMA_updates=2470,late_mean_snapshots=5,
        inherited_component_epochs=80,cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,
        cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,initialization_IQ_examples=0,
        new_calibration_IQ=0,new_trainable_parameters=0,extra_training_neural_forwards=0,external_student_checkpoint_reads=0,
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,R17_training_forward_examples=0,F_training_reference=None)
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('One complete legal source path counted once required')
    if (not math.isfinite(complete.get('seconds',-1)) or complete.get('seconds',-1)<previous
        or not math.isfinite(complete.get('seconds_including_all_readout_serialization',-1))
        or complete.get('seconds_including_all_readout_serialization',-1)<complete['seconds']
        or complete.get('averaging_host_enqueue_seconds')!=host or complete.get('peak_allocated_bytes',0)<=0):raise ValueError('Shared actual training/export cost required')
    losses={}
    for key in ('loss','iq_nmse','spectral_error','teacher_stage_nmse'):
        values=np.asarray([r['losses'][key] for r in records],dtype=np.float64)
        losses[key]=dict(first=float(values[0]),last=float(values[-1]),minimum=float(values.min()),maximum=float(values.max()),mean=float(values.mean()),slope_per_epoch=float(np.polyfit(np.arange(1,21),values,1)[0]))
    return dict(records=records,losses=losses,full_log_records=20,training_complete=complete,
        epoch_seconds_median=float(np.median(np.diff([0.,*[r['seconds'] for r in records]]))),
        source_training_paths=1,cost_counted_once_across_three_readouts=True,
        averaging_time_scope='Host enqueue only;GPU kernels included in actual training seconds;not isolated GPU overhead')


def readout_metadata(info,arm):
    required=dict(readout=arm,rule=d.configuration()['readouts'][arm],shared_source_epochs=20,shared_source_updates=4940,
        ema_updates=2470,late_epoch_snapshots=5,unique_parameters_averaged=714,parameter_aliases=770,buffers_averaged=False,
        teacher_changed=False,extra_neural_forwards=0,new_trainable_parameters=0,TX_reads=0,F_reads=0,V_reads=0,target_reads=0)
    if any(info.get(k)!=v for k,v in required.items()):raise ValueError('Fixed complete source-only parameter readout required')
    return info


def profile_metrics(profile,arm):
    required=dict(status='PROFILE_COMPLETE',includes_full_context_3prior_6CG=True,checkpoint_payload_on_GPU=False,
        optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,physical_GPU=d.GPUS[arm],training_F_access=False,reader_weights_read_for_evaluation_only=True)
    if any(profile.get(k)!=v for k,v in required.items()):raise ValueError('Matched same-card inference profile required')
    sizes=dict(original_R17=4665073,parent_R46_350521=350521,compact=273765)
    if set(profile['results'])!=set(sizes):raise ValueError('All original/parent/readout profiles required')
    cases={m+'_B'+str(b):(b,w,n) for m in ('G_only','G_plus_F') for b,w,n in ((1,16,64),(32,4,16))}
    for name,size in sizes.items():
        model=profile['results'][name]
        if model['parameters']!=size or set(model['cases'])!=set(cases) or model['ConvLinear_MAC_per_IQ']<=0:raise ValueError('Complete profile scope required')
        for key,(batch,warmup,iterations) in cases.items():
            cell=model['cases'][key]
            if (cell['batch'],cell['warmup_per_round'],cell['iterations_per_round'],cell['rounds'],cell['source_V_unique_IQ'])!=(batch,warmup,iterations,3,64):raise ValueError('Three fixed latency rounds required')
            fields=('median_ms','p90_ms','p95_ms','throughput_samples_per_second','resident_before_forward_bytes','peak_allocated_bytes','incremental_peak_bytes')
            if any(not math.isfinite(cell[k]) or cell[k]<=0 for k in fields) or len(cell['round_median_ms'])!=3 or any(not math.isfinite(x) or x<=0 for x in cell['round_median_ms']):raise ValueError('Finite actual inference costs required')
            if not cell['median_ms']<=cell['p90_ms']<=cell['p95_ms'] or cell['peak_allocated_bytes']<cell['resident_before_forward_bytes']:raise ValueError('Latency quantiles or memory scope differs')
    return dict(**profile,latency_ratio_to_R46_by_case={k:profile['results']['compact']['cases'][k]['median_ms']/profile['results']['parent_R46_350521']['cases'][k]['median_ms'] for k in cases})


def completed_metrics(value):
    complete=value.get('complete',{})
    required=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),no_source_score_selection=True,shared_source_training_paths=1,training_counted_once=True)
    if (value.get('run_id')!=d.RUN or value.get('method_commit')!=METHOD or value.get('failure') is not None
        or value.get('actual_live_owned_PIDs') or any(complete.get(k)!=v for k,v in required.items()) or set(value.get('rows',{}))!=set(d.ARMS)):
        raise ValueError('All three formal readouts and owned actual exits required')
    shared=value['shared_source'];finished=shared['completion']
    if (finished.get('status')!='SOURCE_TRAINED' or finished.get('source_worker_exited') is not True
        or finished.get('source_training_counted_once') is not True or finished.get('readouts')!=list(d.ARMS)):
        raise ValueError('Actual shared source child exit before readout prediction required')
    init=shared['initialization']
    expected=dict(mode='original_M35_static_prune_shared_source_path',group_count=4,IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,
        target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False,fixed_original_maps=True,
        parameter_mean_readouts=list(d.ARMS),external_student_checkpoint_reads=0,new_calibration_IQ=0,new_trainable_parameters=0)
    if any(init.get(k)!=v for k,v in expected.items()):raise ValueError('Direct original M35 weight-only source initialization required')
    bases={name:{role:summary_metrics(raw[role],role) for role in ('source','target')} for name,raw in value['baselines'].items()}
    if set(bases)!={'R17','R44','R46'}:raise ValueError('All fixed references required')
    result=dict(status='ANALYZED',run_id=d.RUN,method_commit=METHOD,baselines=bases,rows={},target_feedback=False,goal_complete=False,
        shared_source_training=training_metrics(shared['training_records'],shared['training_complete']),initialization=init,
        limitation='One model seed and one correlated source trajectory,three readouts;human-designated exposed engineering benchmark,not three independent trainings or a new blind test')
    for arm in d.ARMS:
        row=value['rows'][arm]
        if row.get('complete',{}).get('status')!='ANALYZED' or row.get('failure') is not None or 'training_records' in row or 'training_complete' in row:raise ValueError('Frozen readout cannot be counted as another source training')
        metrics={};prediction_cost={}
        for role in ('source','target'):
            expected=dict(status='PREDICTIONS_COMPLETE',role=role,count=1800 if role=='source' else 168000,views=list(d.VIEWS),
                readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED)
            if any(row[role+'_prediction_complete'].get(k)!=v for k,v in expected.items()):raise ValueError('Full same-head seven predictions before truth required')
            timing=row[role+'_prediction_complete'];per_view=timing.get('per_view_seconds',{})
            if (set(per_view)!=set(d.VIEWS) or any(not math.isfinite(x) or x<=0 for x in per_view.values())
                or not math.isfinite(timing.get('seconds',-1)) or timing.get('seconds',-1)<sum(per_view.values())):
                raise ValueError('Separate actual source/formal readout prediction costs required')
            prediction_cost[role]=dict(seconds=timing['seconds'],per_view_seconds=per_view,count=expected['count'],views=7,
                scope='Actual prediction walltime only;excludes profile and independent truth scoring')
            metrics[role]=summary_metrics(row[role+'_summary'],role,training_F_access=False)
        acceptance={name:target_acceptance(metrics['target'],base['target']) for name,base in bases.items()}
        delta={name:compare(metrics['target'],base['target']) for name,base in bases.items()}
        gains={v:delta['R46']['views'][v]['accuracy_delta_pp'] for v in d.VIEWS}
        gaps={v:84.67619047619048-metrics['target']['views'][v]['accuracy'] for v in d.VIEWS if v!='clean'}
        result['rows'][arm]=dict(parameters=273765,context_parameters=185615,training_F_access=False,source_training_shared=True,
            shared_source_ref=d.SOURCE_ARM,reduction_from_R46_percent=100*(1-273765/350521),parameter_readout=readout_metadata(row['parameter_readout'],arm),
            source=metrics['source'],target=metrics['target'],acceptance=acceptance,accepted_both=acceptance['R17']['accepted'] and acceptance['R46']['accepted'],
            target_vs_R17=delta['R17'],target_vs_R44=delta['R44'],target_vs_R46=delta['R46'],profile=profile_metrics(row['matched_profile'],arm),
            checkpoint_bytes=row['checkpoint_bytes'],stdout_findings=row.get('stdout_findings',[]),prediction_cost=prediction_cost,
            performance_optimization=dict(improved_views=[v for v,g in gains.items() if g>0],decreased_views=[v for v,g in gains.items() if g<0],
                low_urban_accuracy=metrics['target']['views']['practical_low_urban']['accuracy'],
                low_urban_at_least_70=metrics['target']['views']['practical_low_urban']['accuracy']>=70,
                other_five_within_5pp_of_raw_clean=all(g<=5 for v,g in gaps.items() if v!='practical_low_urban'),practical_gap_to_raw_clean_pp=gaps,
                acceptance_is_not_optimization_success=True))
    result['same_trajectory_comparisons']={arm:dict(candidate=arm,control='final',
        **{role:compare(result['rows'][arm][role],result['rows']['final'][role]) for role in ('source','target')},
        same_training_path=True,additional_training_examples=0,additional_neural_forwards=0,target_feedback=False) for arm in d.ARMS[1:]}
    return result
