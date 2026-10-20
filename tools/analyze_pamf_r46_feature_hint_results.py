"""Recalculate completed R60 aggregates; no IQ, labels, GPU, or model selection."""
import math
import numpy as np
from experiments.pamf_r46_feature_hint import design as d
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

METHOD='04357684a7c069308e6c89a64d3bb51d54376fdf'


def training_metrics(records,complete,arm):
    if arm not in d.ARMS or len(records)!=20:raise ValueError('All twenty fixed training records required')
    previous=0.
    for epoch,row in enumerate(records,1):
        expected=dict(epoch=epoch,updates=247*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,
            teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,feature_hint_weight=.02,
            teacher_context_feature_vectors=5292000*epoch,teacher_prior_feature_vectors=(5292000*epoch if arm=='all_hint' else 0),teacher_weight=d.teacher_weight(epoch,arm),microbatch=256,source_tensor_backend="direct_float32_buffer")
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Actual source exposure or teacher schedule differs')
        loss=row['losses']
        if set(loss)!={'loss','iq_nmse','spectral_error','teacher_stage_nmse','teacher_context_feature_nmse','teacher_prior_feature_nmse'} or any(not math.isfinite(v) or v<0 for v in loss.values()):
            raise ValueError('Only finite registered F-free losses allowed')
        if not math.isclose(loss['loss'],loss['iq_nmse']+.05*loss['spectral_error']+d.teacher_weight(epoch,arm)*loss['teacher_stage_nmse']+.02*(loss['teacher_context_feature_nmse']+loss['teacher_prior_feature_nmse']),rel_tol=1e-5,abs_tol=1e-7):
            raise ValueError('Weighted objective does not close')
        if arm=='context_hint' and loss['teacher_prior_feature_nmse']!=0:raise ValueError('No prior loss in the context-only row')
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
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,
        feature_hint_scope=arm,feature_hint_weight=.02,context_hint_modules=84,prior_hint_modules=(84 if arm=='all_hint' else 0),
        teacher_context_feature_vectors=105840000,teacher_prior_feature_vectors=(105840000 if arm=='all_hint' else 0),
        extra_teacher_G_forwards=0,extra_student_G_forwards=0,new_calibration_IQ=0,new_trainable_parameters=0,train_only_hooks_removed=True)
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Complete fixed budget or training permissions differ')
    losses={}
    for key in ('loss','iq_nmse','spectral_error','teacher_stage_nmse','teacher_context_feature_nmse','teacher_prior_feature_nmse'):
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
    # Fixed post-freeze comparisons; no IQ, fitting, prediction, ranking or selection.
    for arm,row in value['rows'].items():
        init=row['initialization']
        expected=dict(mode=d.MODE[arm],IQ_examples=0,feature_hint_scope=arm,feature_hint_weight=.02,new_trainable_parameters=0,new_calibration_IQ=0,
            TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,parent_unchanged=True,exact_function_equivalence=False)
        if any(init.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed data-free feature-hint initialization required')
        result['rows'][arm]['feature_hints']=dict(context_modules=84,prior_modules=(84 if arm=='all_hint' else 0),context_weight=.02,
            prior_weight=(.02 if arm=='all_hint' else 0.),energy_floor=1e-4,additional_trainable_parameters=0,new_calibration_IQ=0,
            extra_teacher_G_forwards=0,extra_student_G_forwards=0,train_only_hooks_removed=True,teacher_detached=True,
            teacher_context_feature_vectors=105840000,teacher_prior_feature_vectors=(105840000 if arm=='all_hint' else 0))
    control=value.get('reused_weight_only_control',{})
    if control.get('run')!=d.REUSED_CONTROL['run'] or control.get('row')!='prune_decay' or control.get('method_commit')!=d.REUSED_CONTROL['method_commit']:
        raise ValueError('Fixed R53 weight-only same-width/same-budget control required')
    from tools.analyze_pamf_r46_anchor_refine_results import training_metrics as previous_training
    checked=previous_training(control['training_records'],control['training_complete'],'prune_decay')
    reference={role:summary_metrics(control[role],role) for role in ('source','target')}
    result['reused_weight_only_control']=dict(run=control['run'],row=control['row'],method_commit=control['method_commit'],training_repeated=False,
        parameters=273765,training=checked,**reference)
    result['same_width_hint_comparisons']={a:dict(parameters=273765,feature_control='R53/prune_decay',
        **{role:compare(result['rows'][a][role],reference[role]) for role in ('source','target')},
        graph_initialization_seed_budget_microbatch_IQteacher_matched=True,
        registered_difference='Train-only retained-GN contextfeatures withoptionalprior;extraactivation/reductioncost measured,0extraGforward/params/calibration',target_feedback=False) for a in d.ARMS}
    result['scope_comparison']=dict(candidate='all_hint',control='context_hint',parameters=273765,
        **{role:compare(result['rows']['all_hint'][role],result['rows']['context_hint'][role]) for role in ('source','target')},
        same_context_weight=True,extra_prior_weight=.02,target_feedback=False)
    return result
