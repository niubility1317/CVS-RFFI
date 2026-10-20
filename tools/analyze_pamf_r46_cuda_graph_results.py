"""Independent arithmetic on completed graph-run aggregates; no IQ or truth IO."""
import math
from experiments.pamf_r46_cuda_graph import design as d
from experiments.pamf_r46_prior_preserve.evaluation import compact_acceptance
from tools.analyze_pamf_r17_task_results import summary_metrics,compare
from tools.r17_compact_acceptance import target_acceptance

METHOD='c60a675e4bbc2c207e0a16d36c0ceec075238324'


def profile_metrics(profile,arm):
    required=dict(status='PROFILE_COMPLETE',arm=arm,physical_GPU=d.GPUS[arm],TX_labels_read=0,
                  target_reads=0,state_updates=0,training_F_access=False,includes_full_context_3prior_6CG=True)
    if any(profile.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete same-card frozen-function profile required')
    models={'parent_R46_350521':350521}
    if arm!=d.ARMS[0]:models['candidate']=d.PARAMETERS[arm]
    if set(profile['results'])!=set(models):raise ValueError('Matched parent/candidate profile missing')
    keys={mode+'_B'+str(batch)+'_'+execution for mode in ('G_only','G_plus_F') for batch in (1,32) for execution in ('native','graph')}
    for name,size in models.items():
        value=profile['results'][name]
        if value['parameters']!=size or value['ConvLinear_MAC_per_IQ']<=0 or set(value['cases'])!=keys:
            raise ValueError('Full parameters/MAC or timing matrix differs')
        for key,row in value['cases'].items():
            batch=1 if '_B1_' in key else 32
            expected=dict(batch=batch,rounds=3,warmup_per_round=16 if batch==1 else 4,
                          iterations_per_round=64 if batch==1 else 16,source_V_unique_IQ=64)
            if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Matched timing repetitions differ')
            if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('median_ms','p90_ms','p95_ms','throughput_samples_per_second','peak_allocated_bytes')):
                raise ValueError('Finite positive measured cost required')
            if len(row['round_median_ms'])!=3 or any(not math.isfinite(v) or v<=0 for v in row['round_median_ms']):raise ValueError('All3timing rounds required')
            if not 0<=row['incremental_peak_bytes']<=row['peak_allocated_bytes'] or not 0<=row['resident_before_forward_bytes']<=row['peak_allocated_bytes']:
                raise ValueError('Invalid measured memory scope')
    current=profile['results']['parent_R46_350521' if arm==d.ARMS[0] else 'candidate']['cases']
    parent=profile['results']['parent_R46_350521']['cases'];comparison={}
    for mode in ('G_only','G_plus_F'):
        for batch in (1,32):
            key=mode+'_B'+str(batch)
            a,b,p,q=(m[key+'_'+e] for m,e in ((current,'native'),(current,'graph'),(parent,'native'),(parent,'graph')))
            comparison[key]=dict(candidate_native_to_graph_median_speedup=a['median_ms']/b['median_ms'],
                parent_native_to_candidate_graph_median_speedup=p['median_ms']/b['median_ms'],
                parent_graph_to_candidate_graph_median_speedup=q['median_ms']/b['median_ms'],
                candidate_native_to_graph_throughput_ratio=b['throughput_samples_per_second']/a['throughput_samples_per_second'],
                parent_native_to_candidate_graph_throughput_ratio=b['throughput_samples_per_second']/p['throughput_samples_per_second'],
                parent_graph_to_candidate_graph_throughput_ratio=b['throughput_samples_per_second']/q['throughput_samples_per_second'])
    return dict(raw=profile,comparisons=comparison)


def agreement_metrics(value,role):
    n=1800 if role=='source' else 168000
    expected=dict(status='COMPARED',role=role,count=n,TX_truth_reads=0,native_predictions_reused=True,native_target_reruns=0)
    if role not in ('source','target') or any(value.get(k)!=v for k,v in expected.items()) or set(value['views'])!=set(d.VIEWS):
        raise ValueError('Complete truth-free original-native comparison required')
    for view,row in value['views'].items():
        for key,width in (('iq',512),('features',160),('logits',6)):
            v=row[key];count=v['unequal_elements'];maximum=v['max_abs_difference']
            if type(count) is not int or not 0<=count<=n*width or not math.isfinite(maximum) or maximum<0 or (count==0)!=(maximum==0):
                raise ValueError('Invalid numerical agreement')
        count=row['argmax_disagreements']
        if type(count) is not int or not 0<=count<=min(n,row['logits']['unequal_elements']):raise ValueError('Invalid prediction agreement')
    return dict(**value,all_argmax_equal=all(v['argmax_disagreements']==0 for v in value['views'].values()),
                all_values_bitwise_equal=all(v[k]['unequal_elements']==0 for v in value['views'].values() for k in ('iq','features','logits')))


def completed_metrics(raw):
    if (raw.get('run_id')!=d.RUN or raw.get('method_commit')!=METHOD or raw.get('failure') is not None or raw.get('actual_live_owned_PIDs')
        or raw.get('complete',{}).get('status')!='ANALYZED' or raw['complete'].get('source_rows')!=list(d.ARMS)
        or raw['complete'].get('target_rows')!=list(d.ARMS) or set(raw.get('rows',{}))!=set(d.ARMS)):
        raise ValueError('All three actual rows complete and owned handles exited required')
    bases={name:{role:summary_metrics(v[role],role) for role in ('source','target')} for name,v in raw['baselines'].items()}
    if set(bases)!={'R17','R46'}:raise ValueError('Original and designated references required')
    result=dict(status='ANALYZED',run_id=d.RUN,method_commit=METHOD,baselines=bases,rows={},target_feedback=False,
                long_term_goal_complete=False,limitation='Already exposed single-seed engineering benchmark;fixed weights;no per-detailTX proof')
    for arm,row in raw['rows'].items():
        if row.get('complete',{}).get('status')!='ANALYZED' or row.get('failure') is not None:
            raise ValueError('Full same-row formal scoring required')
        source=row['checkpoint_source']
        if (source['source_arm'],source['epoch'],source['updates'],source['parameters'],source['new_training_updates'],source['training_F_access']) != (d.SOURCE_ROWS[arm],*d.SOURCE_BUDGET[arm],d.PARAMETERS[arm],0,False):
            raise ValueError('Fixed unchanged F-free source checkpoint required')
        smoke=row['smoke']
        if smoke.get('status')!='PASS' or smoke.get('real_Ls_IQ')!=2 or set(smoke['batches'])!={'1','8','32','64','256'} or not smoke['source_checkpoint_weights_unchanged'] or smoke['native_fallback']:
            raise ValueError('Actual-source graph smoke and unchanged weights required')
        metrics={};agreements={};native={}
        for role in ('source','target'):
            manifest=row[role+'_prediction_complete']
            required=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=1800 if role=='source' else 168000,views=list(d.VIEWS),
                readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED,native_fallback=False,batch=256)
            if any(manifest.get(k)!=v for k,v in required.items()):raise ValueError('Complete same-head graph predictions required')
            metrics[role]=summary_metrics(row[role+'_summary'],role,training_F_access=False)
            native[role]=summary_metrics(row['native_'+role+'_summary'],role,training_F_access=False)
            agreements[role]=agreement_metrics(row[role+'_agreement'],role)
            for view in d.VIEWS:
                if agreements[role]['views'][view]['argmax_disagreements']==0 and metrics[role]['views'][view]['confusion']!=native[role]['views'][view]['confusion']:
                    raise ValueError('Identical predictions contradict scored confusion')
        delta={name:compare(metrics['target'],v['target']) for name,v in bases.items()}
        acceptance=dict(R46=compact_acceptance(metrics['target'],bases['R46']['target']),R17=target_acceptance(metrics['target'],bases['R17']['target']))
        gaps={v:84.67619047619047-metrics['target']['views'][v]['accuracy'] for v in d.VIEWS if v!='clean'}
        result['rows'][arm]=dict(parameters=d.PARAMETERS[arm],reduction_from_R46_percent=100*(1-d.PARAMETERS[arm]/350521),new_training_updates=0,training_F_access=False,
            source=metrics['source'],target=metrics['target'],native=native,agreement=agreements,target_vs_R46=delta['R46'],target_vs_R17=delta['R17'],
            target_vs_own_native=compare(metrics['target'],native['target']),acceptance=acceptance,profile=profile_metrics(row['profile'],arm),
            checkpoint_bytes=source['checkpoint_bytes'],standalone_graph_memory=row['standalone_graph_memory'],smoke=smoke,stdout_findings=row.get('stdout_findings',[]),
            performance_optimization=dict(improved_views=[v for v,x in delta['R46']['views'].items() if x['accuracy_delta_pp']>0],
                decreased_views=[v for v,x in delta['R46']['views'].items() if x['accuracy_delta_pp']<0],
                low_urban_at_least_70=metrics['target']['views']['practical_low_urban']['accuracy']>=70,
                other_five_within_5pp_of_raw_clean=all(g<=5 for v,g in gaps.items() if v!='practical_low_urban'),practical_gap_to_raw_clean_pp=gaps))
    result['accepted_rows']=[a for a,r in result['rows'].items() if r['acceptance']['R46']['accepted']]
    result['long_term_goal_complete']=any(r['performance_optimization']['low_urban_at_least_70'] and r['performance_optimization']['other_five_within_5pp_of_raw_clean'] and r['acceptance']['R17']['accepted'] for r in result['rows'].values())
    return result
