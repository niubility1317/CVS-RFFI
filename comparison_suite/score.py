"""Independent truth-last scorer. Never imported by a training/prediction module."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,allow_nan=False)
def csvwrite(path,rows):
    fields=sorted({k for row in rows for k in row})
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def metrics(y,p,classes):
    cm=np.zeros((classes,classes),dtype=np.int64); np.add.at(cm,(y,p),1)
    count=cm.sum(); den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*cm.diagonal(),den,out=np.zeros(classes),where=den!=0)
    support=cm.sum(1); recall=np.divide(cm.diagonal(),support,out=np.zeros(classes),where=support!=0)
    return dict(accuracy=float(cm.diagonal().sum()/count), macro_f1=float(f1.mean()),
                macro_accuracy=float(recall[support>0].mean()), query_count=int(count),confusion=cm.tolist())

def preflight(spec,stage):
    rows=[r for r in spec['rows'] if r['stage']=='phase12']
    if not rows: raise ValueError('No evaluation rows')
    for row in rows:
        flag=read(Path(row['output_root'])/('phase1_complete.json' if stage=='p1' else 'phase2_complete.json'))
        if flag['status']!='PREDICTIONS_COMPLETE' or flag['truth_read']:
            raise ValueError('Predictions incomplete or contaminated; truth remains closed')
    return rows

def p1(spec,rows):
    # Identity, shape and bounds checked for every row before opening truth.
    predictions=[]
    for row in rows:
        cfg=read(row['config']); capsule=Path(cfg['p1_capsule'])
        index=np.load(capsule/'index.npz',allow_pickle=False); manifest=read(capsule/'manifest.json')
        values=np.load(Path(row['output_root'])/'phase1_predictions.npz',allow_pickle=False)
        if not np.array_equal(index['ids'],values['ids']): raise ValueError('P1 identity mismatch')
        for view in ('clean','satellite'):
            pred=values[view]
            if pred.shape!=(len(index['ids']),) or pred.dtype.kind not in 'iu' or pred.min()<0 or pred.max()>=len(manifest['classes']):
                raise ValueError('Invalid P1 predictions')
        predictions.append((row,index,manifest,values))
    truth=read(spec['p1_truth']); results=[]
    for row,index,manifest,values in predictions:
        ids=index['ids'].tolist(); y=np.asarray([truth[s]['label'] for s in ids],dtype=np.int64)
        rx=np.asarray([str(truth[s]['receiver']) for s in ids])
        for view in ['clean','satellite']+manifest['scenes']:
            mask=np.ones(len(ids),dtype=bool) if view in ('clean','satellite') else index['scenes']==manifest['scenes'].index(view)
            pred=values['clean' if view=='clean' else 'satellite']
            for receiver in ['ALL']+sorted(set(rx)):
                use=mask if receiver=='ALL' else mask&(rx==receiver)
                if not use.any(): raise ValueError('Empty P1 stratum')
                results.append(dict(row_id=row['row_id'],method=row['method'],model_seed=row['model_seed'],
                    view=view,receiver=receiver,**metrics(y[use],pred[use],len(manifest['classes']))))
    return results,[]

def validate_record(pred):
    required={'split_id','capsule_id','mode','receiver','scenario','k','support_seed','classes','query_ids','predicted_indices','paired_old_split_id','C_inherits_B','query_fit'}
    if not required.issubset(pred) or pred['mode'] not in {'A_source_frozen','B_old_support','C_native_registration','C_prototype_append_extension','C_knn_append_extension'}:
        raise ValueError('Invalid prediction envelope/mode')
    ids,indices,classes=pred['query_ids'],pred['predicted_indices'],pred['classes']
    if (len(ids)!=len(indices) or not ids or len(set(ids))!=len(ids) or len(set(classes))!=len(classes)
            or pred['query_fit'] or any(type(i)!=int or i<0 or i>=len(classes) for i in indices)):
        raise ValueError('Invalid P2 frozen predictions')
    if pred['C_inherits_B'] != pred['mode'].startswith('C'):
        raise ValueError('Invalid B/C inheritance declaration')

def p2(spec,rows):
    # First pass reads only frozen predictions and markers; all checked before truth.
    capsules={}
    for row in rows:
        cfg=read(row['config']); capsule=Path(cfg['p2_capsule'])
        if str(capsule) not in capsules:
            manifest=read(capsule/'manifest.json')
            splits={s['split_id']:s for s in (read(p) for p in (capsule/'splits').glob('*.json'))}
            with np.load(capsule/'received.npz',allow_pickle=False) as data: ids=data['ids'].copy()
            capsules[str(capsule)]=(manifest,splits,ids)
        manifest,splits,ids=capsules[str(capsule)]
        old_count=len(read(cfg['p1_capsule']+'/manifest.json')['classes'])
        c_mode=('C_prototype_append_extension' if row['method']=='protonet' else 'C_knn_append_extension' if row['method']=='radionet_ada'
                else 'C_native_registration' if row['method'] in {'csil','mopc_hr','orthogonal'} else None)
        expected={(s['split_id'],mode) for s in splits.values() for mode in
            (('A_source_frozen','B_old_support') if len(s['registered_classes'])==old_count else ((c_mode,) if c_mode else ()))}
        count=0
        with (Path(row['output_root'])/'phase2_predictions.jsonl').open(encoding='utf-8') as f:
            seen=set()
            for line in f:
                pred=json.loads(line); validate_record(pred)
                key=(pred['split_id'],pred['mode'])
                if key in seen: raise ValueError('Repeated P2 prediction')
                if key not in expected: raise ValueError('Unregistered P2 mode/split')
                split=splits[pred['split_id']]
                if (pred['capsule_id']!=manifest['capsule_id'] or pred['classes']!=split['registered_classes']
                        or pred['query_ids']!=ids[split['query_indices']].tolist()
                        or any(pred[field]!=split[field] for field in ('receiver','scenario','k','support_seed'))):
                    raise ValueError('Prediction/capsule identity mismatch')
                seen.add(key); count+=1
        if seen!=expected: raise ValueError('Missing registered A/B/C predictions')
        if count!=read(Path(row['output_root'])/'phase2_complete.json')['predictions']:
            raise ValueError('P2 count mismatch')
    truth=read(spec['p2_truth']); results=[]; pairs=[]
    for row in rows:
        old_scores={}; registered=[]
        with (Path(row['output_root'])/'phase2_predictions.jsonl').open(encoding='utf-8') as f:
            for line in f:
                pred=json.loads(line); classes=pred['classes']; targets=[truth[s] for s in pred['query_ids']]
                if any(t['pool_role']!='query' or t['transmitter'] not in classes for t in targets):
                    raise ValueError('Non-query or unregistered scoring identity')
                if any(str(t['receiver'])!=str(pred['receiver']) or t['scenario']!=pred['scenario'] for t in targets):
                    raise ValueError('Query receiver/scenario mismatch')
                y=np.asarray([classes.index(t['transmitter']) for t in targets]); p=np.asarray(pred['predicted_indices'])
                old=np.asarray([bool(t['old']) for t in targets]); ok=(y==p)
                oa=float(ok[old].mean()) if old.any() else None
                na=float(ok[~old].mean()) if (~old).any() else None
                h=2*oa*na/(oa+na) if na is not None and oa+na else (0. if na is not None else None)
                entry=dict(row_id=row['row_id'],method=row['method'],model_seed=row['model_seed'],
                    split_id=pred['split_id'],mode=pred['mode'],receiver=pred['receiver'],scenario=pred['scenario'],
                    k=pred['k'],support_seed=pred['support_seed'],old_class_count=len({t['transmitter'] for t in targets if t['old']}),
                    new_class_count=len({t['transmitter'] for t in targets if not t['old']}),
                    old_accuracy=oa,new_accuracy=na,harmonic_mean=h,**metrics(y,p,len(classes)))
                results.append(entry)
                if pred['mode'].startswith(('A','B')):
                    old_scores[(pred['split_id'],pred['mode'][0])]=(entry,set(pred['query_ids']))
                else: registered.append((pred,entry,{sid for sid,t in zip(pred['query_ids'],targets) if t['old']}))
        for (split_id,stage),(b,ids) in old_scores.items():
            if stage!='B': continue
            a,aids=old_scores[(split_id,'A')]
            if ids!=aids: raise ValueError('A/B query identity mismatch')
            cs=[(pred,c,cids) for pred,c,cids in registered if pred['paired_old_split_id']==split_id]
            cs=[(None,None,ids)]+cs
            for pred,c,cids in cs:
                if cids!=ids: raise ValueError('B/C old physical query mismatch')
                pairs.append(dict(row_id=row['row_id'],method=row['method'],model_seed=row['model_seed'],
                    receiver=b['receiver'],scenario=b['scenario'],k=b['k'],support_seed=b['support_seed'],
                    old_class_count=b['old_class_count'],new_class_count=c['new_class_count'] if c else 0,
                    A_old_accuracy=a['old_accuracy'],B_old_accuracy=b['old_accuracy'],
                    C_old_accuracy=c['old_accuracy'] if c else None,C_new_accuracy=c['new_accuracy'] if c else None,
                    adaptation_gain_pp=100*(b['old_accuracy']-a['old_accuracy']),
                    registration_old_drop_pp=100*(b['old_accuracy']-c['old_accuracy']) if c else None,
                    old_new_gap_pp=100*abs(c['old_accuracy']-c['new_accuracy']) if c else None,
                    H=c['harmonic_mean'] if c else None,C_inherits_B=bool(pred and pred['C_inherits_B']),
                    C_mode=pred['mode'] if pred else ('N/A_no_new_class_registration' if row['method'] in {'protonet','csil','mopc_hr','orthogonal','radionet_ada'} else 'N/A_closed_set_method')))
    return results,pairs

def aggregate(results,stage):
    dimensions=['method','view','receiver'] if stage=='p1' else ['method','mode','scenario','k','new_class_count']
    keys=defaultdict(list)
    for row in results: keys[tuple(row.get(d) for d in dimensions)].append(row)
    summary=[]
    for key,group in keys.items():
        for seed_scope in ('fresh_four','historical_392005','all_five_sensitivity'):
            use=[r for r in group if (seed_scope=='all_five_sensitivity' or (r['model_seed']==392005)==(seed_scope=='historical_392005'))]
            if not use: continue
            seeds=sorted({r['model_seed'] for r in use})
            entry=dict(zip(dimensions,key)); entry.update(seed_scope=seed_scope,model_seeds=seeds,records=len(use))
            for metric in ('accuracy','macro_f1','old_accuracy','new_accuracy','harmonic_mean'):
                means=[float(np.mean([r[metric] for r in use if r['model_seed']==s and r.get(metric) is not None])) for s in seeds if any(r.get(metric) is not None and r['model_seed']==s for r in use)]
                entry[metric+'_mean']=float(np.mean(means)) if means else None
                entry[metric+'_seed_sd']=float(np.std(means,ddof=1)) if len(means)>1 else None
            summary.append(entry)
    return summary

def score(spec,stage):
    root=Path(spec['runtime_root']); marker=root/('scoring_'+stage+'_complete.json')
    if marker.exists(): return read(marker)
    rows=preflight(spec,stage)
    results,paired=p1(spec,rows) if stage=='p1' else p2(spec,rows)
    prefix='phase1' if stage=='p1' else 'phase2'
    write(root/(prefix+'_scored_results.json'),dict(status='SCORED',results=results,target_feedback_forbidden=True))
    csvwrite(root/(prefix+'_scored_results.csv'),[{k:v for k,v in row.items() if k!='confusion'} for row in results])
    write(root/(prefix+'_summary.json'),dict(summary=aggregate(results,stage)))
    if paired: csvwrite(root/'paired_results.csv',paired)
    write(marker,dict(status='SCORED_COMPLETE',records=len(results),paired_records=len(paired),target_feedback_forbidden=True))
    return read(marker)

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--stage',choices=['p1','p2'],required=True)
    a=p.parse_args();print(json.dumps(score(read(a.spec),a.stage)))
if __name__=='__main__':main()
