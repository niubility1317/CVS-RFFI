"""Portable supplementary statistics from frozen CM, source and timing records."""
from collections import defaultdict
from itertools import product
import csv,gzip,json,math,statistics
from pathlib import Path
from .plan import RUN,SEEDS,PROTOCOLS
from .analyze import csvfile

TASKS=('cross_date','cross_rx','cross_both')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban','six_practical_mean')

def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def csvrows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def moment(values):
    a=list(values)
    if len(a)!=3 or not all(math.isfinite(v) for v in a):raise ValueError('Three actual finite seeds required')
    return dict(mean=statistics.mean(a),sd=statistics.stdev(a),positive_seeds=sum(v>0 for v in a),seed_values=';'.join(str(v) for v in a))

def percentile(a,q):
    a=sorted(a);x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)

def supplement(root,out,external_seed=None,external_summary=None):
    root,out=Path(root),Path(out);out.mkdir(exist_ok=True,parents=True)
    values={}
    for r in csvrows(out/'per_seed.csv.gz'):
        if r['dimension']=='overall':values[(r['protocol'],r['method'],int(r['seed']),r['test'],r['readout'],r['view'],r['metric'])]=float(r['percent'])
    external={};external_rows=[]
    if external_seed:
        for r in csvrows(external_seed):
            if r['family']!='pamf' and r['dimension']=='overall':external[(r['protocol'],r['family'],r['cell'],int(r['seed']),r['test'],r['readout'],r['view'],r['metric'])]=float(r['percent'])
    if external_summary:external_rows=[r for r in csvrows(external_summary) if r['family']!='pamf']
    paired=[]
    def pair(pid,name,t,rd,v,m,left,right,note='fixed_G_role'):
        a=[left(s)-right(s) for s in SEEDS]
        paired.append(dict(protocol=pid,effect=name,test=t,readout=rd,view=v,metric=m,interpretation=note,**moment(a)))
    for pid,t,rd,v,m in product(PROTOCOLS,TASKS,('raw','DURA-L','DURA-LU'),VIEWS,('accuracy','macro_f1')):
        def get(method,s,route=rd):return values[pid,method,s,t,route,v,m]
        for control in ('supervised','ordinary_PL'):
            pair(pid,'REPL_minus_'+control,t,rd,v,m,lambda s:get('REPL',s),lambda s:get(control,s))
        if rd!='raw':
            pair(pid,'PL_minus_Sup',t,rd,v,m,lambda s:get('ordinary_PL',s),lambda s:get('supervised',s))
        if external and rd in ('raw','DURA-LU'):
            for family in ('riei','drift','ibrff','sdg'):
                pair(pid,'REPL_minus_'+family+'_D',t,rd,v,m,lambda s:get('REPL',s),lambda s:external[pid,family,'D',s,t,rd,v,m],'complete_recipes_same_contract; no_single_component_causality')
    for pid,t,v,m in product(PROTOCOLS,TASKS,VIEWS,('accuracy','macro_f1')):
        def get(method,s,rd):return values[pid,method,s,t,rd,v,m]
        for method in ('supervised','ordinary_PL','REPL'):
            pair(pid,'LU_minus_L_'+method,t,'LU-L',v,m,lambda s:get(method,s,'DURA-LU'),lambda s:get(method,s,'DURA-L'))
        pair(pid,'PL_by_G_U_interaction',t,'LU-L',v,m,
            lambda s:get('ordinary_PL',s,'DURA-LU')-get('supervised',s,'DURA-LU'),lambda s:get('ordinary_PL',s,'DURA-L')-get('supervised',s,'DURA-L'))
        pair(pid,'matched_REPL_minus_Sup',t,'matched_LU_vs_L',v,m,lambda s:get('REPL',s,'DURA-LU'),lambda s:get('supervised',s,'DURA-L'),'complete_systems; F_and_G_U_permissions_differ')
    matrix=json.loads((root/'inputs/matrix.json').read_text());by={r['row_id']:r for r in matrix['rows']}
    for method in sorted({r['method'] for r in matrix['rows'] if r['group'] in ('mechanism','fine_L','structure')}):
        for t,rd,v,m in product(TASKS,('raw','DURA-L','DURA-LU'),VIEWS,('accuracy','macro_f1')):
            pair('F0','REPL_minus_'+method,t,rd,v,m,lambda s:values['F0','REPL',s,t,rd,v,m],lambda s:values['F0',method,s,t,rd,v,m])
    csvfile(out/'paired_effects.csv',paired)
    main=[]
    for r in csvrows(out/'summary_overall.csv'):
        if r['group']=='main':main.append(dict(protocol=r['protocol'],method='PAMF-'+r['method'],cell='native',test=r['test'],readout=r['readout'],view=r['view'],metric=r['metric'],mean=r['mean'],sd=r['sd']))
    main.extend(dict(protocol=r['protocol'],method=r['family'],cell=r['cell'],test=r['test'],readout=r['readout'],view=r['view'],metric=r['metric'],mean=r['mean'],sd=r['sd']) for r in external_rows)
    csvfile(out/'paper_all_recipes.csv',main)
    costs={};logical=[];counters=[];stage_counter={r['row_id']:r['stages'] for r in records(root/'inputs/stage_counters.jsonl.gz')}
    for item in records(root/'inputs/source_inputs.jsonl.gz'):
        rid=item['row_id'];row=by[rid];done=item['complete'];u_seen=u_accepted=0;L_packets=As_seen=sat_seen=0
        if done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']:raise ValueError('Source freeze/permissions/budget differs')
        for stage in done['segments']:
            folder=stage['folder'];costs[folder]=dict(folder=folder,new_training=stage['config']['run_id']==RUN,epochs=stage['epoch'],updates=stage['updates'],shared_worker_seconds=stage['seconds'])
        for s in stage_counter[rid]:
            p=s['progress']
            if p is None:raise ValueError('Final source counter metadata missing')
            u_seen+=p['PL_seen'];u_accepted+=p['PL_accepted'];L_packets+=p['L_actual']+p['extra_L']
        states=[s['state'] for s in item['REPL_states']]
        if not states and done.get('actual_adapter_SSL',{}).get('hard_PL_tail'):states=[done['actual_adapter_SSL']['hard_PL_tail']]
        tail_seen=sum(s['seen'] for s in states);tail_accept=sum(s['accepted'] for s in states)
        # Final segment progress already includes custom REPL's hidden-U counters.
        # Tail objective states are a cross-check/subset, not additional exposure.
        if states:
            tail_progress=stage_counter[rid][1:]
            if tail_seen!=sum(s['progress']['PL_seen'] for s in tail_progress) or tail_accept!=sum(s['progress']['PL_accepted'] for s in tail_progress):raise ValueError('Tail and final source counters differ')
        if done.get('actual_adapter_SSL') and u_seen!=done['actual_adapter_SSL']['U_seen']:raise ValueError('Historical full U exposure differs')
        if not row['U'] and u_seen:raise ValueError('Supervised row used U')
        for s in states:
            if s['EMA_updates'] or s['U_truth_read'] or s['student_views']!=6*s['seen']:raise ValueError('REPL counters differ from selected recipe')
        logical.append(dict(row_id=rid,protocol=row['protocol_id'],group=row['group'],method=row['method'],seed=row['model_seed'],reused=item['reused'],cumulative_worker_hours=sum(s['seconds'] for s in done['segments'])/3600))
        counters.append(dict(row_id=rid,protocol=row['protocol_id'],method=row['method'],seed=row['model_seed'],U_enabled=row['U'],U_seen=u_seen,U_accepted=u_accepted,
            U_accept_percent=100*u_accepted/u_seen if u_seen else '',REPL_tail_seen=tail_seen,REPL_tail_accepted=tail_accept,
            tail_accept_percent=100*tail_accept/tail_seen if tail_seen else '',L_packet_forward_evaluations=L_packets,
            style_shallow_applied=sum(s['U_style']['applied']['time_down'] for s in states),style_middle_applied=sum(s['U_style']['applied']['t2.pool'] for s in states),
            gradient_guard_conflicts=sum(sum(s.get('U_gradient_guard',{}).get('conflicts',{}).values()) for s in states),
            source_actual_EMA_updates=0,U_truth_read=False,new_inference_parameters=0))
    csvfile(out/'F_unique_cost_replayed.csv',list(costs.values()));csvfile(out/'F_logical_cost_replayed.csv',logical);csvfile(out/'training_exposures.csv',counters)
    new=[v for v in costs.values() if v['new_training']]
    if sum(v['epochs'] for v in new)!=8670 or sum(v['updates'] for v in new)!=1924740:raise ValueError('Deduplicated new training budget differs')
    timing=[];samplecount=0;calls=0
    for panel in records(root/'inputs/timing_inputs.jsonl.gz'):
        c=panel['complete'];groups=defaultdict(list)
        if c['status']!='TIMING_COMPLETE' or not c['exclusive_GPU'] or not c['source_IQ_only'] or len(panel['samples'])!=c['case_rounds']:raise ValueError('Timing panel incomplete')
        for s in panel['samples']:
            if len(s['wall_ms'])!=s['repeats'] or len(s['device_ms'])!=s['repeats'] or len(s['H2D_D2H_ms'])!=20 or not s['exclusive_GPU']:raise ValueError('Timing samples incomplete')
            groups[s['case'],s['batch']].append(s);samplecount+=1;calls+=s['repeats']+s['warmup']+21
        for (case,batch),ss in sorted(groups.items()):
            if {s['round'] for s in ss}!=set(range(5)):raise ValueError('Five actual timing rounds required')
            med=[statistics.median(s['wall_ms']) for s in ss];wall=[v for s in ss for v in s['wall_ms']];trans=[v for s in ss for v in s['H2D_D2H_ms']]
            timing.append(dict(panel=panel['panel'],case=case,batch=batch,rounds=5,wall_median_ms=statistics.median(wall),wall_P95_ms=percentile(wall,.95),round_median_sd=statistics.stdev(med),
                CPU_roundtrip_median_ms=statistics.median(trans),device_median_ms=statistics.median(v for s in ss for v in s['device_ms']),throughput_IQ_s=batch*1000/statistics.median(wall),
                resident_baseline_bytes=max(s['resident_baseline_bytes'] for s in ss),incremental_peak_bytes=max(s['incremental_peak_bytes'] for s in ss),
                execution='optimized_GF' if case.endswith('_GF') else 'eager_F',memory_scope='co_resident; not isolated model VRAM'))
    if samplecount!=1175:raise ValueError('Four complete dedicated timing panels required')
    csvfile(out/'timing_recomputed.csv',timing)
    physics=[];geometry=[];ids=None
    for item in records(root/'inputs/source_physics_inputs.jsonl.gz'):
        d=item['diagnostic'];row=by[item['row_id']]
        if d['samples']!=720 or d['groups']!=90 or d['source_U_reads'] or d['target_reads'] or d['fitted_probe_updates'] or len(d['interventions'])!=11:raise ValueError('Source-only physical diagnostic differs')
        if ids is None:ids=d['physical_ids']
        if d['physical_ids']!=ids:raise ValueError('Source diagnostic sample IDs differ')
        common=dict(method=row['method'],seed=row['model_seed'])
        geometry.append(dict(common,**{k:v for k,v in d['geometry'].items() if k!='pair_rule'}))
        for r in d['interventions']:physics.append(dict(common,**r))
    if len(geometry)!=111:raise ValueError('Complete F0 source-only diagnostics required')
    csvfile(out/'source_geometry.csv',geometry);csvfile(out/'source_interventions.csv',physics)
    # Compare raw confusion matrices for identical F across G component/crossed reads.
    canonical={};extra=[]
    for rec in records(root/'analysis/score_inputs.jsonl.gz'):
        r=rec['row'];raw={tuple(v[k] for k in ('test','view','dimension','stratum')):v['confusion'] for v in rec['score']['results'] if v['readout']=='raw'}
        if r['method']=='REPL' and r['protocol_id'] in ('F0','m3-natural'):canonical[r['protocol_id'],r['model_seed']]=raw
        elif r['group'] in ('G_component','G_crossed'):extra.append((r,raw))
    if len(extra)!=27 or any(raw!=canonical[r['protocol_id'],r['model_seed']] for r,raw in extra):raise ValueError('Same-F raw stratified CMs changed across G readouts')
    verification=dict(status='VERIFIED_SUPPLEMENTARY_INPUTS',new_unique_epochs=8670,new_unique_updates=1924740,new_unique_shared_worker_hours=sum(s['shared_worker_seconds'] for s in new)/3600,
        logical_F=201,new_F=189,reused_F=12,source_diagnostic_models=111,source_diagnostic_packets_per_model=720,source_diagnostic_strata=90,source_interventions=11,
        raw_stratified_CM_reuse_checked_rows=27,timing_case_rounds=samplecount,timing_GPU_calls=calls,external_summary_reused_not_rescored=bool(external_summary),new_G=0,new_IQ_cache=0)
    (out/'supplement_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    return verification
