"""Model-free scorer: every fixed prediction is checked before target truth opens."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from comparison_suite.score import metrics
from experiments.cvs_clean_eval.score import write,csvwrite
from experiments.cvs_all_frozen_clean_backfill.contracts import read,validate_spec,SEEDS,CAPSULE

def preflight_predictions(spec):
    validate_spec(spec); predictions=[]
    capsule=Path(CAPSULE);manifest=read(capsule/'manifest.json')
    if manifest['status']!='VALIDATED_ONCE' or len(manifest['classes'])!=6 or manifest['channel']!='residual/post_sync/noeq':raise ValueError('Wrong validated clean capsule')
    with np.load(capsule/'index.npz',allow_pickle=False) as f:ids=f['ids'].copy()
    if len(ids)!=168000 or len(set(ids.tolist()))!=len(ids):raise ValueError('Wrong physical query IDs')
    for row in spec['rows']:
        root=Path(row['output_root']);c=read(row['config']);resolved=read(root/'resolved_config.json')
        flag=read(root/'clean_complete.json');provenance=read(root/'provenance.json')
        if (flag['status']!='PREDICTIONS_COMPLETE' or flag['count']!=len(ids) or flag['views']!=['clean']
            or flag['truth_read'] is not False or flag['query_fit'] is not False or c['p1_capsule']!=CAPSULE
            or resolved['truth_read'] is not False or resolved['query_fit'] is not False
            or any(resolved.get(k)!=v for k,v in c.items()) or provenance['status']!='VERIFIED'
            or provenance['query_fit'] is not False or provenance['checkpoint']!=row['source_output']+'/last.pt'):
            raise ValueError('Incomplete or mismatched frozen prediction: '+row['row_id'])
        if row['reuse'] and read(root.parents[1]/'scoring_clean_complete.json')['status']!='SCORED_COMPLETE':raise ValueError('Original prediction was not completely scored')
        with np.load(root/'clean_predictions.npz',allow_pickle=False) as f:
            if set(f.files)!={'ids','clean'} or not np.array_equal(f['ids'],ids):raise ValueError('Prediction ID/view differs')
            pred=f['clean'].copy()
        if pred.shape!=(len(ids),) or pred.dtype.kind not in 'iu' or pred.min()<0 or pred.max()>=6:raise ValueError('Invalid class predictions')
        predictions.append((row,pred))
    return ids,predictions

def score(spec):
    root=Path(spec['runtime_root'])
    if (root/'scoring_clean_complete.json').exists():raise FileExistsError('Preserve already-scored backfill')
    ids,predictions=preflight_predictions(spec)
    # This is the first target-truth access; no training/model imports in this process.
    truth=read(spec['p1_truth']);targets=[truth[i] for i in ids.tolist()]
    if any(type(t['label']) is not int or not 0<=t['label']<6 for t in targets):raise ValueError('Invalid truth labels')
    y=np.array([t['label'] for t in targets]);rx=np.array([str(t['receiver']) for t in targets])
    groups=[('ALL',np.ones(len(y),dtype=bool))]+[('RX'+v,rx==v) for v in sorted(set(rx))]
    txgroups=[('TX'+str(v),y==v) for v in range(6)]
    results=[];class_results=[]
    for row,pred in predictions:
        base={k:row[k] for k in ('row_id','model_id','variant','source_run','model_seed')}
        for group,mask in groups:
            results.append(dict(base,group=group,view='clean',**metrics(y[mask],pred[mask],6)))
        for group,mask in txgroups:
            class_results.append(dict(base,group=group,count=int(mask.sum()),accuracy=float((pred[mask]==y[mask]).mean())))
    bygroup=defaultdict(list)
    for r in results:bygroup[(r['model_id'],r['group'])].append(r)
    summary=[]
    for (model_id,group),rows in bygroup.items():
        q=dict(model_id=model_id,variant=rows[0]['variant'],source_run=rows[0]['source_run'],group=group,
               model_seeds=sorted(SEEDS),query_count_per_seed=rows[0]['query_count'])
        for m in ('accuracy','macro_f1','macro_accuracy'):
            q[m+'_mean']=float(np.mean([r[m] for r in rows]));q[m+'_seed_sd']=float(np.std([r[m] for r in rows],ddof=1))
        summary.append(q)
    lookup={(r['model_id'],r['group'],r['model_seed']):r for r in results}
    controls=[m for m in {r['model_id'] for r in results} if next(r['variant'] for r in results if r['model_id']==m) in ('native','residual_fusion','relation_frequency_energy')]
    paired=[]
    for model_id in sorted({r['model_id'] for r in results}):
        for baseline in sorted(controls):
            if model_id==baseline:continue
            for group,_ in groups:
                ds=[100*(lookup[(model_id,group,s)]['accuracy']-lookup[(baseline,group,s)]['accuracy']) for s in sorted(SEEDS)]
                paired.append(dict(model_id=model_id,baseline=baseline,group=group,delta_pp_by_seed=ds,delta_pp_mean=float(np.mean(ds)),delta_pp_sd=float(np.std(ds,ddof=1)),positive_seeds=sum(v>0 for v in ds)))
    write(root/'clean_scored_results.json',dict(status='SCORED',results=results,class_results=class_results,target_feedback_forbidden=True))
    write(root/'clean_summary.json',dict(summary=summary,paired=paired,target_feedback_forbidden=True,scope='Retrospective fixed benchmark;not blind confirmation or source reselection'))
    csvwrite(root/'clean_scored_results.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    csvwrite(root/'clean_summary.csv',summary);csvwrite(root/'clean_class_results.csv',class_results)
    csvwrite(root/'clean_paired.csv',paired)
    marker=dict(status='SCORED_COMPLETE',models=76,seeds=4,rows=304,new_predictions=216,reused_predictions=88,
                records=len(results),view='clean',query_count=len(ids),decisions=len(ids)*304,target_feedback_forbidden=True)
    write(root/'scoring_clean_complete.json',marker)
    return marker

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(score(read(a.spec)))
