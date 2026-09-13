"""Merge disjoint, independently verified target batches without re-scoring truth."""
import json
import csv
import statistics as st
from pathlib import Path
import argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',nargs='+',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    batches=[json.loads(Path(p).read_text(encoding='utf-8')) for p in a.inputs]
    d=json.loads(json.dumps(batches[0]))
    d['batches']=[{'input':p,'complete':b['complete'],'manifest':b['manifest']} for p,b in zip(a.inputs,batches)]
    for b in batches[1:]:
        assert not set(d['scores'])&set(b['scores']), 'Overlapping model batches'
        for key in ('target_rxs','target_days','samples_per_scene','scenes','batch_size','augmentation_seed','num_classes','target_exposure_status'):
            assert d['manifest'][key]==b['manifest'][key], key
        for key in ('scores','source_scores','verification','confusion_matrices'):
            d[key].update(b[key])
        for key in ('training_seeds','checkpoints'):
            d['manifest'][key].update(b['manifest'][key])
        d['manifest']['rows']+=b['manifest']['rows']
        d['paired_predictions']+=b['paired_predictions']
    d['complete']={'complete':all(b['complete']['complete'] for b in batches),'rows':d['manifest']['rows'],'total_seconds':sum(b['complete']['total_seconds'] for b in batches)}
    assert len(d['scores'])==21 and sum(v['predictions'] for v in d['verification'].values())==14112000
    d['manifest']['all_rows_predicted_before_truth']=False
    d['manifest']['all_rows_predicted_before_truth_within_each_batch']=True
    d.pop('predictions_complete',None)
    d.pop('stdout',None)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
    pairs=[]
    for base,var in [('A','B'),('A','C'),('B','D'),('A','E'),('B','F'),('E','F')]:
        for seed in (392005,392006,392007):
            x=d['scores'][f'V2_{base}_seed{seed}'];y=d['scores'][f'V2_{var}_seed{seed}']
            pairs.append(dict(comparison=f'{var}-{base}',seed=seed,**{s+'_delta_pp':100*(y['scenes'][s]['accuracy']-x['scenes'][s]['accuracy']) for s in d['manifest']['scenes']},leo_delta_pp=100*(y['leo_mean_accuracy']-x['leo_mean_accuracy'])))
    with (out.parent/'target_all_paired_metric_deltas.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps({'models':len(d['scores']),'delta_means':{p:st.mean(r['leo_delta_pp'] for r in pairs if r['comparison']==p) for p in sorted({r['comparison'] for r in pairs})}}))

if __name__=='__main__':main()
