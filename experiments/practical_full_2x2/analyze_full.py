"""Reproduce the 627 frozen readouts from serialized CMs, with no truth/model IO."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict,Counter
from contextlib import ExitStack
from pathlib import Path
from experiments.cric_dura_comparison.analyze_main import metrics,moments,write_csv,SEEDS,VIEWS
from .snapshot_results import TASKS,VARIANTS,validate_panel


def read(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text(encoding='utf-8'))


def run(folder,output):
    folder,out=Path(folder),Path(output);out.mkdir(parents=True,exist_ok=False)
    manifest=read(folder/'manifest.json');plan=manifest['matrix']
    registered={r['row_id']:r for r in plan['rows']+plan['reused_rows']}
    new_ids={r['row_id'] for r in plan['rows']}
    prior=read(folder/manifest['prior_overall_snapshot'])
    reused={r['row']['row_id']:r['score'] for r in prior['rows']}
    supplement=read(folder/'cost_supplement.json.gz')
    historical={r['model_seed']:r for r in supplement['historical_F0_F']}
    summary=[];gain_summary=[];effects=[];cost=[];stages=[];gcost=[];cache_cost=[]
    seen=set();CM_new=CM_reused=net_new=net_reused=0;physical={};prediction_seconds=0.
    with ExitStack() as stack:
        writers={}
        def emit(name,row):
            if name not in writers:
                stream=stack.enter_context(gzip.open(out/(name+'.csv.gz'),'wt',encoding='utf-8',newline=''))
                writers[name]=csv.DictWriter(stream,fieldnames=list(row));writers[name].writeheader()
            writers[name].writerow(row)
        for filename in manifest['protocol_files']:
            obj=read(folder/filename);protocol=obj['protocol'];records=obj['rows']
            for r in records:r['row']=registered[r['row_id']]
            validate_panel(records);values={};population_base=None
            for rec in records:
                row=rec['row'];rid=row['row_id'];family=row['family'];cell=row['cell'];seed=row['model_seed']
                if rid in seen:raise ValueError('Duplicate logical source')
                seen.add(rid);score=rec['score'];reader='DURA-'+row['G_roles'];classes=len(rec['classes'])
                if classes!=(10 if protocol=='ManyRx' else 6):raise ValueError('Registered class count differs')
                pred=rec['prediction_complete'];done=rec['score_complete']
                if (score['row_id']!=rid or score['model_seed']!=seed or not score['truth_last']
                    or not score['target_feedback_forbidden'] or pred['truth_read'] or pred['query_fit']
                    or pred['status']!='ALL_PREDICTIONS_COMPLETE' or done['status']!='SCORED_COMPLETE'
                    or done['views']!=list(VIEWS) or done['readouts']!=['raw',reader]):
                    raise ValueError('Prediction/score identity, order or coverage differs')
                prediction_seconds+=pred['seconds']
                old={(r['test'],r['readout'],r['view'],r['dimension'],str(r['stratum'])):r
                     for r in reused.get(rid,{}).get('results',[])}
                actual={};counts={}
                for r in score['results']:
                    key=(r['test'],r['readout'],r['view'],r['dimension'],str(r['stratum']))
                    if key in actual:raise ValueError('Duplicate score cell')
                    actual[key]=r
                    if key in old:
                        if r!=old[key]:raise ValueError('Previously delivered overall score changed')
                        val={m:r[m] for m in ('accuracy','macro_f1')};CM_reused+=1
                    else:val=metrics(r,classes);CM_new+=1
                    values[(family,cell,seed,*key)]=val
                    if r['readout']=='raw' and r['view']=='clean':counts[(r['test'],r['dimension'],str(r['stratum']))]=r['count']
                if population_base is None:population_base=counts
                elif counts!=population_base:raise ValueError('Matched physical population differs')
                required={(t,route,v,dim,s) for t,dim,s in counts for route in ('raw',reader) for v in VIEWS}
                if set(actual)!=required:raise ValueError('Missing route, task, view or physical stratum')
                for t in TASKS:physical[protocol,t]=counts[t,'overall','ALL']
                pairs={(r['test'],r['readout'],r['view'],r['dimension'],str(r['stratum'])):r for r in score['paired_raw']}
                if set(pairs)!={k for k in required if k[1]!='raw'}:raise ValueError('Incomplete restoration net cells')
                oldpairs={(r['test'],r['readout'],r['view'],r['dimension'],str(r['stratum'])):r
                          for r in reused.get(rid,{}).get('paired_raw',[])}
                for key,p in pairs.items():
                    t,route,v,dim,s=key;raw=actual[t,'raw',v,dim,s];restored=actual[key]
                    if key in oldpairs:
                        if p!=oldpairs[key]:raise ValueError('Delivered net correction changed')
                        net_reused+=1
                    else:
                        diag=lambda r:sum(r['confusion'][i][i] for i in range(classes))
                        net=diag(restored)-diag(raw);n=raw['count'];correct=diag(raw)
                        if (p['net']!=net or p['count']!=n or p['repairs']-p['damage']!=net
                            or min(p['repairs'],p['damage'])<0 or p['repairs']>n-correct or p['damage']>correct):
                            raise ValueError('Net corrections differ from confusion matrices')
                        net_new+=1
                    emit('net_corrections',dict(protocol=protocol,family=family,cell=cell,seed=seed,
                        test=t,readout=route,view=v,dimension=dim,stratum=s,count=p['count'],
                        repairs=p['repairs'],damage=p['damage'],net=p['net']))
                meta=rec['source_metadata'];seconds=updates=0;epochs=0;pl_seen=pl_accepted=sat_seen=sat_applied=None
                if row['kind']=='external':
                    freeze=meta['freeze.json']
                    if freeze['target_access'] or freeze['target_score_used'] or not freeze['default_head']:
                        raise ValueError('External source permissions differ')
                    segments=freeze['stages'];seconds=freeze['seconds']
                    pl_seen=freeze['pl_seen'];pl_accepted=freeze['pl_accepted']
                    sat_seen=freeze['satellite_examples'];sat_applied=freeze['satellite_applied_examples']
                    if ((cell in 'AC' and pl_seen!=0) or (cell in 'BD' and pl_seen<=0)
                        or (cell in 'AB' and sat_applied!=0) or (cell in 'CD' and sat_applied<=0)
                        or pl_accepted>pl_seen):raise ValueError('Actual PL/augmentation branch differs')
                    for seg in segments:
                        updates+=seg['updates'];epochs+=seg['epochs'] or 0
                        stages.append(dict(row_id=rid,protocol=protocol,family=family,cell=cell,seed=seed,
                            stage=seg['stage'],epochs=seg['epochs'],updates=seg['updates'],
                            optimizer_steps=seg['optimizer_steps'],shared_worker_seconds=seg['seconds'],
                            pl_seen=seg['pl_seen'],pl_accepted=seg['pl_accepted'],
                            sat_seen=seg['satellite_examples'],sat_applied=seg['satellite_applied_examples'],new_F=True))
                elif meta:
                    complete=meta['complete.json'];segments=complete['segments'];seconds=complete['seconds']
                    if complete['target_access'] or complete['U_truth_read'] or complete['updates']!=11100:
                        raise ValueError('PAMF source budget/permissions differ')
                    progress=[v for k,v in meta.items() if k.endswith('/progress.json')]
                    if cell in 'AC' and any(v['U_actual'] for v in progress):raise ValueError('Supervised PAMF consumed U')
                    for i,seg in enumerate(segments):
                        updates+=seg['updates'];epochs+=seg['epoch']
                        stages.append(dict(row_id=rid,protocol=protocol,family=family,cell=cell,seed=seed,
                            stage=i+1,epochs=seg['epoch'],updates=seg['updates'],optimizer_steps=seg['updates'],
                            shared_worker_seconds=seg['seconds'],pl_seen=None,pl_accepted=None,
                            sat_seen=None,sat_applied=None,new_F=rid in new_ids))
                else:
                    if protocol!='F0' or cell!='native_T1':raise ValueError('Unknown historical cost source')
                    for seg in historical[seed]['stages']:
                        updates+=seg['updates'];epochs+=seg['epochs'];seconds+=seg['seconds']
                        stages.append(dict(row_id=rid,protocol=protocol,family=family,cell=cell,seed=seed,
                            stage=seg['name'],epochs=seg['epochs'],updates=seg['updates'],optimizer_steps=seg['updates'],
                            shared_worker_seconds=seg['seconds'],pl_seen=None,pl_accepted=None,
                            sat_seen=None,sat_applied=None,new_F=False))
                cost.append(dict(row_id=rid,protocol=protocol,family=family,cell=cell,seed=seed,new_F=rid in new_ids,
                    epochs=epochs,updates=updates,shared_worker_seconds=seconds,pl_seen=pl_seen,pl_accepted=pl_accepted,
                    pl_acceptance=pl_accepted/pl_seen if pl_seen else None,sat_seen=sat_seen,sat_applied=sat_applied,
                    sat_rate=sat_applied/sat_seen if sat_seen else None,
                    source_key=rec['source_output'] or 'original_T1_'+str(seed)))
            for key in list(values):
                if key[5]!='clean':continue
                fam,cell,seed,t,route,_,dim,s=key
                six=[values[(fam,cell,seed,t,route,v,dim,s)] for v in VIEWS[1:]]
                values[(fam,cell,seed,t,route,'six_practical_mean',dim,s)]={m:statistics.mean(r[m] for r in six) for m in ('accuracy','macro_f1')}
            grouped=defaultdict(dict);gains=defaultdict(dict)
            for k,val in sorted(values.items()):
                fam,cell,seed,t,route,v,dim,s=k
                for metric,value in val.items():
                    rec=dict(protocol=protocol,family=fam,cell=cell,seed=seed,test=t,readout=route,view=v,
                             dimension=dim,stratum=s,metric=metric,percent=100*value)
                    emit('per_seed',rec)
                    grouped[(fam,cell,t,route,v,dim,s,metric)][seed]=100*value
                    if route!='raw':gains[(fam,cell,t,route,v,dim,s,metric)][seed]=100*(value-values[(fam,cell,seed,t,'raw',v,dim,s)][metric])
            fields=('family','cell','test','readout','view','dimension','stratum','metric')
            for groups,name,target in ((grouped,'summary',summary),(gains,'restoration_gain',gain_summary)):
                for k,v in sorted(groups.items()):
                    rec=dict(protocol=protocol,**dict(zip(fields,k)),**moments([v[s] for s in SEEDS]))
                    emit(name,rec)
                    if rec['dimension']=='overall':target.append(rec)
            for fam in ('riei','drift','ibrff','sdg'):
                for t in TASKS:
                    for view in (*VIEWS,'six_practical_mean'):
                        for metric in ('accuracy','macro_f1'):
                            for effect,coeffs in [('channel_no_PL',{'C':1,'A':-1}),('channel_with_PL',{'D':1,'B':-1}),
                                ('PL_no_channel',{'B':1,'A':-1}),('PL_with_channel',{'D':1,'C':-1}),
                                ('interaction',{'D':1,'C':-1,'B':-1,'A':1})]:
                                vals=[100*sum(w*values[(fam,c,seed,t,'raw',view,'overall','ALL')][metric] for c,w in coeffs.items()) for seed in SEEDS]
                                effects.append(dict(protocol=protocol,family=fam,effect=effect,test=t,view=view,metric=metric,**moments(vals)))
            for cache in obj['caches']:
                complete=cache['complete'];known=next((v for v in cache['G_metadata'].values() if 'seconds_by_stage' in v),None)
                cache_cost.append(dict(protocol=protocol,readout=cache['role'],seconds=complete['seconds'],
                    cached_packets=sum(complete['counts'].values())*7,new_G_training=False))
                if known and (len(known['seconds_by_stage'])!=8 or known['updates']!=39520
                        or any(known[k] for k in ('TX_labels_read','F_reads','target_reads'))):raise ValueError('G ancestry incomplete or permissions differ')
                gcost.append(dict(protocol=protocol,readout=cache['role'],
                    new_G=protocol in ('F1_rx_repartition_r01','F2_rx_repartition_r01'),
                    known_full_eight_stage_cost=bool(known),shared_worker_seconds=sum(known['seconds_by_stage'].values()) if known else None,
                    updates=39520 if known else None,checkpoint=cache['G']['checkpoint'],
                    missing_ancestry_cost='Original fixed F0 G-LU: historical ancestor timings not fully serialized' if not known else ''))
            print(json.dumps(dict(protocol=protocol,rows=len(records),new_CMs=CM_new,reused_overall_CMs=CM_reused)),flush=True)
    if seen!=set(registered) or len(cost)!=627 or sum(r['new_F'] for r in cost)!=600:raise ValueError('Incomplete final matrix')
    for name,rows in [('summary_overall',summary),('restoration_gain_overall',gain_summary),('factorial_raw_overall',effects),
        ('F_cost',cost),('F_stage_cost',stages),('G_cost',gcost),('restored_cache_cost',cache_cost)]:write_csv(out/(name+'.csv'),rows)
    aug=supplement['augmentation_cache']
    if any(r[k] for r in aug for k in ('U_reads','V_reads','target_reads','labels_used_for_transform')):raise ValueError('Augmentation producer permissions differ')
    write_csv(out/'augmentation_producer_cost.csv',[{k:v for k,v in r.items() if k!='scene_counts'}|{'scene_counts':json.dumps(r['scene_counts'],sort_keys=True)} for r in aug])
    for r in aug:
        if sum(r['scene_counts'].values())!=r['count']:raise ValueError('Augmentation scene counts differ')
    verification=dict(status='ANALYZED_FULL627',logical_models=627,new_F=600,reused_F=27,new_G=4,
        protocols=list(plan['protocols']),F_seeds=list(SEEDS),all_predictions_frozen=True,independent_scores=627,
        new_serialized_CM_recomputed=CM_new,reused_verified_overall_CM=CM_reused,
        new_net_cells_recomputed=net_new,reused_verified_overall_nets=net_reused,
        total_serialized_CM=CM_new+CM_reused,total_net_cells=net_new+net_reused,
        new_F_shared_worker_hours=sum(r['shared_worker_seconds'] for r in cost if r['new_F'])/3600,
        reused_F_historical_shared_worker_hours=sum(r['shared_worker_seconds'] for r in cost if not r['new_F'])/3600,
        new_G_shared_worker_hours=sum(r['shared_worker_seconds'] for r in gcost if r['new_G'])/3600,
        known_reused_G_shared_worker_hours=sum(r['shared_worker_seconds'] or 0 for r in gcost if not r['new_G'])/3600,
        G_historical_cost_missing=[r['protocol']+'/'+r['readout'] for r in gcost if not r['known_full_eight_stage_cost']],
        augmentation_cache_records=len(aug),augmentation_production_elapsed_hours=sum(r['seconds'] for r in aug)/3600,
        restored_IQ_cache_elapsed_hours=sum(r['seconds'] for r in cache_cost)/3600,
        cached_F_prediction_elapsed_hours=prediction_seconds/3600,
        external_zero_PL_acceptance_rows=sum(r['family']!='pamf' and r['pl_seen'] and r['pl_accepted']==0 for r in cost),
        unknown_PAMF_selected_gate_acceptance=True,unknown_PAMF_per_F_realized_channel_counts=True,
        physical_populations=[dict(protocol=p,test=t,count=n) for (p,t),n in physical.items()],
        F_standard_deviation_conditional_on_fixed_G=True,not_significance_test=True,target_truth_reads=0,target_feedback_used=False)
    (out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.input,args.output)
