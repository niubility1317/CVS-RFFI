"""Independent arithmetic on completed R28 aggregates; no IQ, F, or truth IO."""
import math
from experiments.pamf_r46_prior_collapse import design as d
from experiments.pamf_r46_prior_preserve.evaluation import compact_acceptance
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.analyze_pamf_r46_cuda_graph_results import agreement_metrics
from tools.r17_compact_acceptance import target_acceptance

METHOD='7ca5af4f6f5d9f862e07e13b9da15902379044e4'


def collapse_metrics(value):
    expected=dict(mode='prior_only_linear_pair_collapse',before_parameters=278180,after_parameters=299297,
        factor_modules_replaced=84,unique_kernel_pairs=28,context_unchanged=True,normalization_unchanged=True,
        source_model_unchanged=True,new_training_updates=0,IQ_reads=0,F_reads=0,TX_labels_read=0)
    if any(value.get(k)!=v for k,v in expected.items()) or not math.isfinite(value.get('seconds',-1)) or value.get('seconds',-1)<0:
        raise ValueError('Frozen prior-only complete conversion required')
    return value


def smoke_metrics(value,arm):
    required=dict(parameters=299297,source_parameters=278180,strict_reload_exact=True,source_unchanged=True,new_training_updates=0,training_F_access=False)
    if any(value.get(k)!=v for k,v in required.items()) or set(value.get('views',{}))!={'clean','received01'}:
        raise ValueError('Actual source conversion smoke required')
    collapse_metrics(value['conversion'])
    for view in value['views'].values():
        if (not view.get('restorer_close') or not view.get('strict_reload_exact') or view.get('F_equivalence_gate') is not False
            or type(view.get('argmax_disagreements')) is not int or not 0<=view['argmax_disagreements']<=2
            or set(view['outputs'])!={'IQ','variance','features','logits'}):
            raise ValueError('Restorer equivalence and descriptive F boundary required')
        for key,record in view['outputs'].items():
            if any(not math.isfinite(record[k]) or record[k]<0 for k in ('max_abs','relative_L2')) or type(record.get('elementwise_close')) is not bool:
                raise ValueError('Finite source output diagnostics required')
            if key in ('IQ','variance') and not record['elementwise_close']:raise ValueError('Original IQ/variance tolerance must pass')
    return value


def profile_metrics(value,arm):
    required=dict(status='PROFILE_COMPLETE',physical_GPU=d.GPUS[arm],includes_full_context_3prior_6CG=True,
        checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,training_F_access=False)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Measured full-physics same-card profile required')
    sizes=dict(original_R17=4665073,parent_R46_350521=350521,lowrank_R22=278180,compact=299297)
    if set(value['results'])!=set(sizes):raise ValueError('All four matched models required')
    keys={mode+'_B'+str(batch) for mode in ('G_only','G_plus_F') for batch in (1,32)}
    for name,size in sizes.items():
        model=value['results'][name]
        if model['parameters']!=size or model['ConvLinear_MAC_per_IQ']<=0 or set(model['cases'])!=keys:raise ValueError('Complete deployed parameters and costs required')
        for key,row in model['cases'].items():
            batch=1 if key.endswith('_B1') else 32
            expected=dict(batch=batch,rounds=3,warmup_per_round=16 if batch==1 else 4,iterations_per_round=64 if batch==1 else 16,source_V_unique_IQ=64)
            if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Matched timing repetitions differ')
            if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('median_ms','p90_ms','p95_ms','throughput_samples_per_second','peak_allocated_bytes')):
                raise ValueError('Finite positive measured cost required')
            if len(row['round_median_ms'])!=3 or any(not math.isfinite(v) or v<=0 for v in row['round_median_ms']):raise ValueError('All three timing rounds required')
            if not 0<=row['incremental_peak_bytes']<=row['peak_allocated_bytes'] or not 0<=row['resident_before_forward_bytes']<=row['peak_allocated_bytes']:
                raise ValueError('Measured memory scope invalid')
    if value['results']['compact']['ConvLinear_MAC_per_IQ']<=value['results']['lowrank_R22']['ConvLinear_MAC_per_IQ']:
        raise ValueError('Expanded dense execution increases measured MAC;cannot hide it')
    comparisons={}
    for key,cur in value['results']['compact']['cases'].items():
        comparisons[key]={label:dict(median_speedup=base['cases'][key]['median_ms']/cur['median_ms'],
            latency_change_percent=100*(cur['median_ms']/base['cases'][key]['median_ms']-1),
            throughput_ratio=cur['throughput_samples_per_second']/base['cases'][key]['throughput_samples_per_second'])
            for label,base in [('vs_parent_R46',value['results']['parent_R46_350521']),('vs_lowrank_R22',value['results']['lowrank_R22'])]}
    return dict(raw=value,comparisons=comparisons)


def completed_metrics(raw):
    if (raw.get('run_id')!=d.RUN or raw.get('method_commit')!=METHOD or raw.get('failure') is not None or raw.get('actual_live_owned_PIDs')
        or raw.get('complete',{}).get('status')!='ANALYZED' or raw['complete'].get('source_rows')!=list(d.ARMS)
        or raw['complete'].get('target_rows')!=list(d.ARMS) or set(raw.get('rows',{}))!=set(d.ARMS)):
        raise ValueError('Both complete rows and actual owned handles exited required')
    smoke=raw['controller_smoke'];required=dict(status='PASS',real_L_s_IQ=2,source_views=['clean','received01'],TX_labels_read=0,V_reads=0,target_reads=0,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0)
    if any(smoke.get(k)!=v for k,v in required.items()) or set(smoke['arms'])!=set(d.ARMS):raise ValueError('Actual two-Ls noquery smoke required')
    bases={name:{role:summary_metrics(v[role],role) for role in ('source','target')} for name,v in raw['baselines'].items()}
    if set(bases)!={'R17','R46'}:raise ValueError('Original and designated references required')
    result=dict(status='ANALYZED',run_id=d.RUN,method_commit=METHOD,baselines=bases,rows={},target_feedback=False,long_term_goal_complete=False,
        limitation='Fixed both R22 source-final inference reparameterizations;zero new training;single exposed engineeringbenchmark;not TXdetail proof')
    for arm,row in raw['rows'].items():
        if row.get('failure') is not None or row.get('complete',{}).get('status')!='ANALYZED' or not row['complete'].get('all_predictions_before_truth') or row['complete'].get('new_training_updates')!=0:
            raise ValueError('Full same-row truth-last scoring required')
        origin=row['checkpoint_source'];required=dict(source_run=d.COMPLEX_RUN,source_row=d.SOURCE_ROWS[arm],source_commit=d.COMPLEX_COMMIT,
            inherited_epoch=20,inherited_updates=19700,new_training_updates=0,source_parameters=278180,parameters=299297)
        if any(origin.get(k)!=v for k,v in required.items()):raise ValueError('Fixed frozen source-final lineage required')
        if row['runtime_ready']!=dict(status='FROZEN_RUNTIME_READY',arm=arm,parameters=299297,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0):
            raise ValueError('Complete expanded runtime required')
        audit=collapse_metrics(row['collapse']);checked_smoke=smoke_metrics(smoke['arms'][arm],arm)
        if audit!=checked_smoke['conversion']:raise ValueError('Delivered source conversion differs from smoke export')
        metrics={};native={};agreements={}
        for role in ('source','target'):
            expected=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=1800 if role=='source' else 168000,views=list(d.VIEWS),readouts=['restored'],
                TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED,original_default_head=True,native_fallback=False,batch=128,execution='prior_collapsed_native_FP32',setup=[])
            if any(row[role+'_prediction_complete'].get(k)!=v for k,v in expected.items()):raise ValueError('Same-head full seven collapsed predictions required')
            metrics[role]=summary_metrics(row[role+'_summary'],role,training_F_access=False)
            native[role]=summary_metrics(row['native_'+role+'_summary'],role,training_F_access=False)
            agreements[role]=agreement_metrics(row[role+'_agreement'],role)
            for view,v in agreements[role]['views'].items():
                if v['argmax_disagreements']==0 and metrics[role]['views'][view]['confusion']!=native[role]['views'][view]['confusion']:
                    raise ValueError('Identical predictions contradict scored confusion')
        delta={name:compare(metrics['target'],v['target']) for name,v in bases.items()}
        acceptance=dict(R46=compact_acceptance(metrics['target'],bases['R46']['target']),R17=target_acceptance(metrics['target'],bases['R17']['target']))
        gap={v:84.67619047619047-metrics['target']['views'][v]['accuracy'] for v in d.VIEWS if v!='clean'}
        result['rows'][arm]=dict(parameters=299297,source_parameters=278180,context_parameters=186703,prior_parameters=112594,
            reduction_from_R46_percent=100*(1-299297/350521),new_training_updates=0,training_F_access=False,collapse=audit,smoke=checked_smoke,
            source=metrics['source'],target=metrics['target'],native=native,agreement=agreements,target_vs_R46=delta['R46'],target_vs_R17=delta['R17'],
            target_vs_own_native=compare(metrics['target'],native['target']),acceptance=acceptance,profile=profile_metrics(row['profile'],arm),
            checkpoint_bytes=row['checkpoint_bytes'],stdout_findings=row.get('stdout_findings',[]),performance_optimization=dict(
                low_urban_at_least_70=metrics['target']['views']['practical_low_urban']['accuracy']>=70,
                other_five_within_5pp_of_raw_clean=all(g<=5 for v,g in gap.items() if v!='practical_low_urban'),
                clean_preserved_vs_raw=metrics['target']['views']['clean']['net_repaired']>=0,practical_gap_to_raw_clean_pp=gap,acceptance_is_not_optimization_success=True))
    result['accepted_rows']=[a for a,r in result['rows'].items() if r['acceptance']['R46']['accepted']]
    result['numeric_target_rows']=[a for a,r in result['rows'].items() if all(r['performance_optimization'][k] for k in ('low_urban_at_least_70','other_five_within_5pp_of_raw_clean','clean_preserved_vs_raw')) and all(v['accepted'] for v in r['acceptance'].values())]
    return result
