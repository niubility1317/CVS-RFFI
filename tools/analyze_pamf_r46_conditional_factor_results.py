"""Recalculate completed R18 artifacts, without IQ/labels or model selection."""
import math
import numpy as np
from experiments.pamf_r46_conditional_factor import design as d
from experiments.pamf_r46_conditional_factor.evaluation import compact_acceptance
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

METHOD='aca99499d3f0170df71380f6429ac9a01ef412f8'

def training_metrics(records,complete,arm):
    if arm not in d.ARMS or len(records)!=20:raise ValueError('All20 fixed source records required')
    keys={'loss','iq_nmse','spectral_error','teacher_stage_nmse','conditional_prior_nmse'}
    previous=0.
    for epoch,row in enumerate(records,1):
        expected=dict(epoch=epoch,updates=985*epoch,L_exposures=6300*epoch,U_exposures=56700*epoch,teacher_IQ_examples=63000*epoch,source_IQ_examples=63000*epoch,F_inputs=0,teacher_weight=d.teacher_weight(epoch))
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Actual source exposure/teacher schedule differs')
        losses=row['losses']
        if set(losses)!=keys or any(not math.isfinite(v) or v<0 for v in losses.values()):raise ValueError('Finite registered F-free losses required')
        total=losses['iq_nmse']+.05*losses['spectral_error']+d.teacher_weight(epoch)*losses['teacher_stage_nmse']+d.CONDITIONAL_WEIGHT[arm]*losses['conditional_prior_nmse']
        if not math.isclose(losses['loss'],total,rel_tol=1e-5,abs_tol=1e-7):raise ValueError('Weighted objective does not close')
        if arm=='factor625_stage' and losses['conditional_prior_nmse']!=0:raise ValueError('Control must skip conditional branch')
        if not math.isfinite(row['seconds']) or row['seconds']<=previous or not 0<=row['clip_fraction']<=1:raise ValueError('Invalid time or clipping')
        if not math.isclose(row['learning_rate_last'],d.learning_rate(985*epoch-1),rel_tol=1e-9):raise ValueError('LR differs')
        previous=row['seconds']
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=19700,L_exposures=126000,U_exposures=1134000,source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,
        parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm],context_parameters=d.CONTEXT_PARAMETERS[arm],context_updates=19700,coarse_updates=19700,action_updates=19700,prior_updates=19700,context_trainable=True,
        cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,initialization_IQ_examples=0,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,V_reads=0,target_reads=0,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,
        conditional_weight=d.CONDITIONAL_WEIGHT[arm],conditional_teacher_prior_examples=3*1260000*int(d.CONDITIONAL_WEIGHT[arm]>0))
    if any(complete.get(k)!=v for k,v in expected.items()):raise ValueError('Complete source budget or permissions differ')
    curves={}
    for key in keys:
        values=np.asarray([r['losses'][key] for r in records])
        curves[key]=dict(first=float(values[0]),last=float(values[-1]),minimum=float(values.min()),maximum=float(values.max()),mean=float(values.mean()),slope_per_epoch=float(np.polyfit(np.arange(1,21),values,1)[0]))
    return dict(full_log_records=20,records=records,losses=curves,training_complete=complete,epoch_seconds_median=float(np.median(np.diff([0.,*[r['seconds'] for r in records]]))),checkpoint_selection='fixed final E20/19700')

def profile_metrics(profile,arm):
    expected=dict(status='PROFILE_COMPLETE',includes_full_context_3prior_6CG=True,checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,training_F_access=False,physical_GPU=d.GPUS[arm])
    if any(profile.get(k)!=v for k,v in expected.items()):raise ValueError('Actual matched complete chain profile required')
    sizes=dict(original_R17=4665073,parent_R46_350521=350521,compact=278180)
    if set(profile['results'])!=set(sizes):raise ValueError('All matched baselines required')
    for name,size in sizes.items():
        value=profile['results'][name]
        if value['parameters']!=size or set(value['cases'])!={'G_only_B1','G_only_B32','G_plus_F_B1','G_plus_F_B32'}:raise ValueError('Full cost scope differs')
        for key,row in value['cases'].items():
            batch=1 if key.endswith('_B1') else 32
            if (row['batch'],row['rounds'],row['iterations_per_round'],row['warmup_per_round'],row['source_V_unique_IQ'])!=(batch,3,64 if batch==1 else 16,16 if batch==1 else 4,64):raise ValueError('Matched repetitions differ')
            if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('median_ms','p90_ms','p95_ms','throughput_samples_per_second','peak_allocated_bytes')):raise ValueError('Finite measured cost required')
    return profile

def completed_metrics(value):
    if (value.get('run_id')!=d.RUN or value.get('method_commit')!=METHOD or value.get('failure') is not None or value.get('actual_live_owned_PIDs')
        or value.get('complete',{}).get('status')!='ANALYZED' or value['complete'].get('source_rows')!=list(d.ARMS) or value['complete'].get('target_rows')!=list(d.ARMS) or set(value.get('rows',{}))!=set(d.ARMS)):
        raise ValueError('All actual rows complete and owned handles exited required')
    bases={name:{role:summary_metrics(raw[role],role) for role in ('source','target')} for name,raw in value['baselines'].items()}
    if set(bases)!={'R17','R46'}:raise ValueError('Original and designated references required')
    result=dict(status='ANALYZED',run_id=d.RUN,method_commit=METHOD,baselines=bases,rows={},target_feedback=False,long_term_goal_complete=False,limitation='Single seed;human-designated exposed engineering benchmark;not independent blind validation')
    for arm,row in value['rows'].items():
        if row.get('complete',{}).get('status')!='ANALYZED' or row.get('failure') is not None:raise ValueError('Complete same-row scoring required')
        metrics={}
        for role in ('source','target'):
            expected=dict(status='PREDICTIONS_COMPLETE',role=role,count=1800 if role=='source' else 168000,views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED)
            if any(row[role+'_prediction_complete'].get(k)!=v for k,v in expected.items()):raise ValueError('Same-head whole-seven predictions required')
            metrics[role]=summary_metrics(row[role+'_summary'],role,training_F_access=False)
        acceptance=dict(R46=compact_acceptance(metrics['target'],bases['R46']['target']),R17=target_acceptance(metrics['target'],bases['R17']['target']))
        deltas={name:compare(metrics['target'],base['target']) for name,base in bases.items()};gains={v:deltas['R46']['views'][v]['accuracy_delta_pp'] for v in d.VIEWS}
        gap={v:84.67619047619047-metrics['target']['views'][v]['accuracy'] for v in d.VIEWS if v!='clean'}
        result['rows'][arm]=dict(parameters=d.PARAMETERS[arm],context_parameters=d.CONTEXT_PARAMETERS[arm],reduction_from_R46_percent=100*(1-d.PARAMETERS[arm]/350521),training_F_access=False,
            training=training_metrics(row['training_records'],row['training_complete'],arm),source=metrics['source'],target=metrics['target'],target_vs_R46=deltas['R46'],target_vs_R17=deltas['R17'],acceptance=acceptance,
            profile=profile_metrics(row['matched_profile'],arm),checkpoint_bytes=row['checkpoint_bytes'],stdout_findings=row.get('stdout_findings',[]),
            performance_optimization=dict(improved_views=[v for v,g in gains.items() if g>0],decreased_views=[v for v,g in gains.items() if g<0],low_urban_accuracy=metrics['target']['views']['practical_low_urban']['accuracy'],
                low_urban_at_least_70=metrics['target']['views']['practical_low_urban']['accuracy']>=70,other_five_within_5pp_of_raw_clean=all(g<=5 for v,g in gap.items() if v!='practical_low_urban'),practical_gap_to_raw_clean_pp=gap,acceptance_is_not_optimization_success=True))
    result['accepted_rows']=[a for a,r in result['rows'].items() if r['acceptance']['R46']['accepted']]
    result['long_term_goal_complete']=any(r['performance_optimization']['low_urban_at_least_70'] and r['performance_optimization']['other_five_within_5pp_of_raw_clean'] and r['acceptance']['R17']['accepted'] for r in result['rows'].values())
    result['matched_conditional_vs_stage']=compare(result['rows']['factor625_cond']['target'],result['rows']['factor625_stage']['target'])
    return result
