"""Independent model-free scoring, all predictions before first truth access."""
import argparse,csv
from pathlib import Path
import numpy as np
from comparison_suite.score import metrics
from experiments.cvs_reference_residual_clean.contracts import read,write,validate_spec,validate_config,SEEDS,VARIANTS,CAPSULE,CLASSES,QUERY_COUNT,SOURCE_COMMIT

OUTPUTS=('clean_scored_results.json','clean_summary.json','clean_scored_results.csv','clean_summary.csv','clean_class_results.csv','clean_paired.csv','scoring_clean_complete.json')
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def preflight_predictions(spec):
    validate_spec(spec);manifest=read(Path(CAPSULE)/'manifest.json')
    if manifest.get('status')!='VALIDATED_ONCE' or manifest.get('classes')!=CLASSES or manifest.get('channel')!='residual/post_sync/noeq':raise ValueError('Clean capsule differs')
    with np.load(Path(CAPSULE)/'index.npz',allow_pickle=False) as f:ids=f['ids'].copy()
    if len(ids)!=QUERY_COUNT or len(set(ids.tolist()))!=QUERY_COUNT:raise ValueError('Physical test IDs differ')
    predictions=[]
    for row in spec['rows']:
        root=Path(row['output_root']);c=validate_config(read(row['config']))
        flag=read(root/'clean_complete.json');resolved=read(root/'resolved_config.json');proof=read(root/'provenance.json')
        if (any(flag.get(k)!=v for k,v in dict(status='PREDICTIONS_COMPLETE',count=QUERY_COUNT,views=['clean'],truth_read=False,query_fit=False).items())
            or any(resolved.get(k)!=v for k,v in c.items()) or resolved.get('truth_read') is not False or resolved.get('query_fit') is not False
            or proof.get('status')!='VERIFIED' or proof.get('query_fit') is not False or proof.get('checkpoint')!=row['source_output']+'/last.pt'):
            raise ValueError('Incomplete or contaminated prediction; truth remains closed')
        with np.load(root/'clean_predictions.npz',allow_pickle=False) as f:
            if set(f.files)!={'ids','clean'} or not np.array_equal(f['ids'],ids):raise ValueError('Prediction ID/view mismatch')
            pred=f['clean'].copy()
        if pred.shape!=(QUERY_COUNT,) or pred.dtype.kind not in 'iu' or pred.min()<0 or pred.max()>=6:raise ValueError('Invalid predicted classes')
        predictions.append((row,pred))
    return ids,predictions

def score(spec):
    root=Path(spec['runtime_root'])
    for name in OUTPUTS:
        if (root/name).exists():raise FileExistsError('Preserve scoring evidence: '+name)
    ids,predictions=preflight_predictions(spec)
    # The first target truth read occurs here, after all eight fixed arrays pass.
    truth=read(spec['p1_truth']);target=[truth[i] for i in ids.tolist()]
    if any(type(t['label']) is not int or not 0<=t['label']<6 for t in target):raise ValueError('Truth class mismatch')
    y=np.asarray([t['label'] for t in target]);rx=np.asarray([str(t['receiver']) for t in target])
    if set(rx)!={'1-1','14-7','2-1','20-1','7-14','7-7','8-8'}:raise ValueError('Expected seven test receivers')
    groups=[('ALL',np.ones(len(y),dtype=bool))]+[('RX'+v,rx==v) for v in sorted(set(rx))]
    results=[];classes=[]
    for row,pred in predictions:
        base={k:row[k] for k in ('row_id','variant','model_seed')}
        for group,mask in groups:results.append(dict(base,group=group,view='clean',**metrics(y[mask],pred[mask],6)))
        for label in range(6):
            mask=y==label;classes.append(dict(base,tx=CLASSES[label],count=int(mask.sum()),accuracy=float((y[mask]==pred[mask]).mean())))
    summary=[];lookup={(r['variant'],r['group'],r['model_seed']):r for r in results}
    for variant in VARIANTS:
        for group,_ in groups:
            q=dict(variant=variant,group=group,view='clean',model_seeds=list(SEEDS),query_count_per_seed=lookup[(variant,group,SEEDS[0])]['query_count'])
            for metric in ('accuracy','macro_f1','macro_accuracy'):
                values=[lookup[(variant,group,s)][metric] for s in SEEDS]
                q[metric+'_mean']=float(np.mean(values));q[metric+'_seed_sd']=float(np.std(values,ddof=1))
            summary.append(q)
    paired=[]
    for group,_ in groups:
        delta=[100*(lookup[('reference_residual_inverse',group,s)]['accuracy']-lookup[('reference_residual_scalar',group,s)]['accuracy']) for s in SEEDS]
        paired.append(dict(group=group,inverse_minus_scalar_pp=delta,mean_pp=float(np.mean(delta)),sd_pp=float(np.std(delta,ddof=1)),positive_seeds=sum(v>0 for v in delta)))
    write(root/'clean_scored_results.json',dict(status='SCORED',results=results,class_results=classes,target_feedback_forbidden=True))
    write(root/'clean_summary.json',dict(summary=summary,paired=paired,target_feedback_forbidden=True,source_commit=SOURCE_COMMIT))
    csvwrite(root/'clean_scored_results.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    csvwrite(root/'clean_summary.csv',summary);csvwrite(root/'clean_class_results.csv',classes);csvwrite(root/'clean_paired.csv',paired)
    done=dict(status='SCORED_COMPLETE',models=2,seeds=4,rows=8,records=len(results),class_records=len(classes),view='clean',query_count=QUERY_COUNT,decisions=8*QUERY_COUNT,target_feedback_forbidden=True)
    write(root/'scoring_clean_complete.json',done);return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(score(read(a.spec)))
