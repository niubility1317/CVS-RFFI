"""Recalculate completed R65 aggregates; no IQ, labels, GPU, or model selection."""
import math
import numpy as np
from experiments.pamf_r46_delayed_prune import design as d
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

METHOD='4fd058f9bae79687f924e06d057e6a20083ff6dc'


def training_metrics(records,complete,arm):
    if arm not in d.ARMS or len(records)!=20:raise ValueError('All twenty fixed training records required')
    previous=0.
    for epoch,row in enumerate(records,1):
        expected=dict(epoch=epoch,updates=247*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,
            teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,teacher_weight=d.teacher_weight(epoch),microbatch=256,phase=d.phase(epoch,arm),parameters=(350521 if d.phase(epoch,arm)=='dense' else 273765),optimizer_resets=int(epoch>d.WARMUP_EPOCHS[arm]),source_tensor_backend="direct_float32_buffer")
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
        cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,
        cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,initialization_IQ_examples=0,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,prior_parameters_changed=True,new_calibration_IQ=0,new_trainable_parameters=0,external_student_checkpoint_reads=0,
        dense_epochs=d.WARMUP_EPOCHS[arm],compact_epochs=20-d.WARMUP_EPOCHS[arm],dense_IQ_examples=63000*d.WARMUP_EPOCHS[arm],
        compact_IQ_examples=63000*(20-d.WARMUP_EPOCHS[arm]),optimizer_resets=1,channel_maps_unchanged=True,global_schedule_restarted=False)
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Complete fixed budget or training permissions differ')
    if set(complete.get('phase_seconds',{}))!={'dense','compact'} or any(not math.isfinite(v) or v<=0 for v in complete['phase_seconds'].values()):raise ValueError('Actual dense and compact training costs required')
    losses={}
    for key in ('loss','iq_nmse','spectral_error','teacher_stage_nmse'):
        values=np.asarray([r['losses'][key] for r in records],dtype=np.float64)
        losses[key]=dict(first=float(values[0]),last=float(values[-1]),minimum=float(values.min()),maximum=float(values.max()),
            mean=float(values.mean()),slope_per_epoch=float(np.polyfit(np.arange(1,21),values,1)[0]))
    return dict(full_log_records=20,actual_microbatch=256,source_tensor_backend="direct_float32_buffer",records=records,losses=losses,training_complete=complete,
        epoch_seconds_median=float(np.median(np.diff([0.,*[r['seconds'] for r in records]]))),checkpoint_selection='fixed final E20/4940')


def profile_metrics(profile,arm):
    expected=dict(status='PROFILE_COMPLETE',includes_full_context_3prior_6CG=True,checkpoint_payload_on_GPU=False,
        optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,physical_GPU=d.GPUS[arm],training_F_access=False,
        reader_weights_read_for_evaluation_only=True)
    if any(profile.get(k)!=v for k,v in expected.items()):raise ValueError('Matched same-GPU repeated G/GplusF cost required')
    sizes=dict(original_R17=4665073,parent_R46_350521=350521,compact=d.PARAMETERS[arm])
    if set(profile['results'])!=set(sizes):raise ValueError('All original,parent,compact profiles required')
    cases={m+'_B'+str(b):(b,w,n) for m in ('G_only','G_plus_F') for b,w,n in ((1,16,64),(32,4,16))}
    for name,size in sizes.items():
        item=profile['results'][name]
        if item['parameters']!=size or set(item['cases'])!=set(cases) or item['ConvLinear_MAC_per_IQ']<=0:raise ValueError('Complete cost scope differs')
        for key,(batch,warmup,iterations) in cases.items():
            cell=item['cases'][key]
            if (cell['batch'],cell['warmup_per_round'],cell['iterations_per_round'],cell['rounds'],cell['source_V_unique_IQ'])!=(batch,warmup,iterations,3,64):raise ValueError('Fixed repeated timing count differs')
            if len(cell['round_median_ms'])!=3 or any(not math.isfinite(v) or v<=0 for v in cell['round_median_ms']):raise ValueError('Finite repeated latency required')
            for metric in ('median_ms','p90_ms','p95_ms','throughput_samples_per_second','resident_before_forward_bytes','peak_allocated_bytes','incremental_peak_bytes'):
                if not math.isfinite(cell[metric]) or cell[metric]<=0:raise ValueError('Finite actual cost required')
            if not cell['median_ms']<=cell['p90_ms']<=cell['p95_ms'] or cell['peak_allocated_bytes']<cell['resident_before_forward_bytes']:raise ValueError('Cost quantiles or peak scope differ')
    ratios={key:profile['results']['compact']['cases'][key]['median_ms']/profile['results']['parent_R46_350521']['cases'][key]['median_ms'] for key in cases}
    return dict(**profile,latency_ratio_to_R46_by_case=ratios,
        limitation='64 source V unique IQ,repeated correlated timings,preloaded-device synchronization;threeG/oneF share resident memory;not hardware-independent end-to-end service latency')


def completed_metrics(value):
    if (value.get('run_id')!=d.RUN or value.get('method_commit')!=METHOD or value.get('failure') is not None
            or value.get('actual_live_owned_PIDs') or value.get('complete',{}).get('status')!='ANALYZED'
            or value['complete'].get('source_rows')!=list(d.ARMS) or value['complete'].get('target_rows')!=list(d.ARMS)
            or set(value.get('rows',{}))!=set(d.ARMS)):
        raise ValueError('Both actual fixed rows completed and owned handles exited required')
    bases={name:{role:summary_metrics(raw[role],role) for role in ('source','target')} for name,raw in value['baselines'].items()}
    if set(bases)!={'R17','R44','R46'}:raise ValueError('Both original references required')
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
        acceptance['R44']['reference']='R44_442805_report_only'
        acceptance['R46']['reference']='human_R46_350521'
        deltas={name:compare(metrics['target'],base['target']) for name,base in bases.items()}
        gains={view:deltas['R46']['views'][view]['accuracy_delta_pp'] for view in d.VIEWS}
        gap={view:84.67619047619047-metrics['target']['views'][view]['accuracy'] for view in d.VIEWS if view!='clean'}
        result['rows'][arm]=dict(parameters=d.PARAMETERS[arm],context_parameters=d.CONTEXT_PARAMETERS[arm],
            reduction_from_R44_percent=100*(1-d.PARAMETERS[arm]/442805),reduction_from_R46_percent=100*(1-d.PARAMETERS[arm]/350521),training_F_access=False,
            training=training_metrics(row['training_records'],row['training_complete'],arm),source=metrics['source'],target=metrics['target'],
            target_vs_R17=deltas['R17'],target_vs_R46=deltas['R46'],target_vs_R44=deltas['R44'],acceptance=acceptance,
            accepted_both=acceptance['R17']['accepted'] and acceptance['R46']['accepted'],profile=profile_metrics(row['matched_profile'],arm),
            checkpoint_bytes=row['checkpoint_bytes'],stdout_findings=row.get('stdout_findings',[]),
            performance_optimization=dict(improved_views=[v for v,g in gains.items() if g>0],decreased_views=[v for v,g in gains.items() if g<0],
                low_urban_accuracy=metrics['target']['views']['practical_low_urban']['accuracy'],
                low_urban_at_least_70=metrics['target']['views']['practical_low_urban']['accuracy']>=70,
                other_five_within_5pp_of_raw_clean=all(g<=5 for v,g in gap.items() if v!='practical_low_urban'),practical_gap_to_raw_clean_pp=gap,
                acceptance_is_not_optimization_success=True))
    # Fixed maps and actual source transition are checked without reading IQ or weights.
    for arm,row in value['rows'].items():
        init=row['initialization'];warm=d.WARMUP_EPOCHS[arm]
        expected=dict(mode='dense_M35_then_fixed_map_pruning',phase='dense',warmup_epochs=warm,initial_parameters=350521,
            IQ_examples=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,
            parent_unchanged=True,exact_initial_function_equivalence=True,map_recomputed_after_initialization=False,teacher='original_M35_only')
        if any(init.get(k)!=v for k,v in expected.items()):raise ValueError('Original dense M35 initialization differs')
        maps=init.get('channel_maps',{})
        if set(maps)!={'coarse_spectral','coarse_temporal','action','prior0','prior1','prior2'}:raise ValueError('All six core fixed maps required')
        for name,levels in maps.items():
            old=(8,12,20,32) if name=='coarse_spectral' else (12,20,32,44)
            new=(8,12,16,28) if name=='coarse_spectral' else (12,16,28,40)
            if len(levels)!=4:raise ValueError('Four resolutions required')
            for indices,width,narrow in zip(levels,old,new):
                if (len(indices)!=narrow or indices!=sorted(set(indices)) or any(type(v) is not int or v<0 or v>=width for v in indices)
                    or any(sum(g*(width//4)<=v<(g+1)*(width//4) for v in indices)!=narrow//4 for g in range(4))):raise ValueError('Original GN4 compatible fixed maps required')
        if any(maps['prior0']!=maps['prior'+str(i)] for i in (1,2)):raise ValueError('Tied prior maps differ')
        transition=row.get('transition',{})
        expected=dict(status='SOURCE_TRANSITION_COMPLETE',completed_epoch=warm,completed_updates=warm*247,
            L_exposures=warm*6300,U_exposures=warm*56700,optimizer_resets=1,initial_parameters=350521,parameters=273765,
            channel_maps_unchanged=True,teacher_replaced=False,teacher_parameter_updates=0,
            external_student_checkpoint_reads=0,new_calibration_IQ=0,global_schedule_restarted=False)
        if any(transition.get(k)!=v for k,v in expected.items()) or not math.isfinite(transition.get('seconds',-1)) or transition.get('seconds',-1)<0:
            raise ValueError('Actual complete source-only transition required')
        result['rows'][arm]['fixed_map_initialization']=init;result['rows'][arm]['source_transition']=transition
    if value['rows']['prune_after5']['initialization']['channel_maps']!=value['rows']['prune_after10']['initialization']['channel_maps']:
        raise ValueError('Both schedules require the same original maps')
    controls=value.get('reused_controls',{})
    if set(controls)!={'static','retain'}:raise ValueError('Both preregistered historical controls required')
    for name,expected in [('static',d.STATIC_CONTROL),('retain',d.RETAIN_CONTROL)]:
        control=controls[name]
        if any(control.get(k)!=expected[k] for k in ('run','row','commit')):raise ValueError('Historical control identity differs')
        trained=control['training_complete']
        required=dict(epochs=20,updates=4940,parameters=expected['parameters'],L_exposures=126000,U_exposures=1134000,
            TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0)
        if any(trained.get(k)!=v for k,v in required.items()):raise ValueError('Historical control source contract differs')
        reference={role:summary_metrics(control[role],role) for role in ('source','target')}
        result['reused_'+name+'_control']=dict(run=control['run'],row=control['row'],method_commit=control['commit'],
            parameters=expected['parameters'],training_repeated=False,**reference)
        key='static_budget_comparisons' if name=='static' else 'matched_budget_comparisons'
        result[key]={a:dict(parameters=273765,control=control['run']+'/'+control['row'],
            **{role:compare(result['rows'][a][role],reference[role]) for role in ('source','target')},
            matched_parent_seed_E20_updates_microbatch_IQteacher=True,optimizer_reset_and_actual_compute_differ=True,target_feedback=False) for a in d.ARMS}
    result['timing_comparison']=dict(candidate='prune_after5',control='prune_after10',parameters=273765,
        **{role:compare(result['rows']['prune_after5'][role],result['rows']['prune_after10'][role]) for role in ('source','target')},
        initial_weights_maps_global_budget_equal=True,dense_compact_compute_and_reset_time_differ=True,target_feedback=False)
    return result
