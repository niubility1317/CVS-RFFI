"""Paired source L -> single source V diagnostics; never target feedback."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from cvs_d92_matched import fit_d92, load_ground, predict_scores, read
from cvsrffi.stage2_d92_support_cv import fit_support_cv


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path)
    a=p.parse_args();spec=read(a.spec);cfg=spec['development'];out=Path(cfg['evaluation_output'])
    if out.exists():raise FileExistsError(out)
    marker=read(Path(cfg['feature_output'])/'complete.json')
    if marker['status']!='SOURCE_FEATURES_COMPLETE' or marker['target_access'] is not False:raise ValueError('Source export incomplete')
    ground=load_ground(cfg['ground'])
    if read(Path(cfg['ground'])/'manifest.json')['checkpoint_sha256']!=marker['checkpoint_sha256']:raise ValueError('Ground provenance mismatch')
    data=np.load(Path(cfg['feature_output'])/'features.npz',allow_pickle=False)
    x,y,logits=data['features'],data['labels'],data['logits'];classes=marker['classes']
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    results=[]
    with (out/'metrics.jsonl').open('x',encoding='utf-8') as stream:
        for rx in sorted(set(data['receivers'].tolist())):
            for scene in range(3):
                cell=(data['receivers']==rx)&(data['scenes']==scene)
                source=[np.flatnonzero(cell&(data['roles']=='L_s')&(y==c)) for c in range(6)]
                valid=[np.flatnonzero(cell&(data['roles']=='V')&(y==c)) for c in range(6)]
                rng=np.random.default_rng(np.random.SeedSequence([cfg['support_seed'],int(rx),scene]))
                support=[rng.permutation(v) for v in source]
                query=np.concatenate([v[:cfg['validation_per_class']] for v in valid])
                if any(len(v)<max(cfg['k']) for v in source) or any(len(v)<cfg['validation_per_class'] for v in valid):
                    raise ValueError('Insufficient source cell coverage; do not silently omit a row')
                for k in cfg['k']:
                    s=np.concatenate([v[:k] for v in support])
                    for method in cfg['methods']:
                        started=time.perf_counter()
                        if method=='frozen_dg':
                            score=logits[query];audit={}
                        elif method=='D92':
                            state=fit_d92(support_features=x[s],support_labels=y[s],classes=classes,ground=ground,seed=cfg['model_seed'])
                            score=predict_scores(state,x[query]);audit=dict(active_feature_dim=state.active_feature_dim)
                        else:
                            state=fit_support_cv(support_features=x[s],support_labels=y[s],support_logits=logits[s],classes=classes)
                            score=state.score(x[query],logits[query]);audit=state.audit
                        prediction=score.argmax(1)
                        accuracy=float(np.mean(prediction==y[query]))
                        record=dict(receiver=int(rx),scene=cfg['scenarios'][scene],k=k,method=method,
                            accuracy=accuracy,validation_count=len(query),elapsed_seconds=time.perf_counter()-started,
                            new_accuracy=None,harmonic=None,metric_scope='source_six_seen_TX_adaptation_only',audit=audit)
                        stream.write(json.dumps(record)+'\n');stream.flush();results.append(record)
                        print(json.dumps({key:value for key,value in record.items() if key!='audit'}),flush=True)
    with (out/'summary.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','k','accuracy','cells']);writer.writeheader()
        for method in cfg['methods']:
            for k in cfg['k']:
                rows=[r for r in results if r['method']==method and r['k']==k]
                writer.writerow(dict(method=method,k=k,accuracy=float(np.mean([r['accuracy'] for r in rows])),cells=len(rows)))
    (out/'complete.json').write_text(json.dumps(dict(status='SOURCE_DIAGNOSTIC_COMPLETE',rows=len(results),
        target_access=False,source_validation_role='single_V',new_class_generalization_verified=False),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
