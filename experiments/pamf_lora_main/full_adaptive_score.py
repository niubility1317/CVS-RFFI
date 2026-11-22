"""Score all frozen full LoRa seeds on the same 30,000 target physical rows."""
import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import shlex
from statistics import mean, stdev
import subprocess
import tarfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .collect import wait_for_predictions
from .data import CLASSES, TARGET, VIEWS, label_ids, packet_indices, recording
from .full_adaptive import EPOCHS, METHODS, SET_RUN, WARMUP_METHODS, WARMUP_RUN, case, cases, methods_for
from .full_packet_data import LIMIT
from .full_supervised import FULL_EPOCHS, SEEDS
from .packet_score import score_refined_u
from .scoring import macro_f1


def matrix(root,methods=METHODS):
    preds={}
    for name,seed,method in cases(methods):
        p=root/'predictions'/name
        if read(p/'complete.json')!=dict(status='PREDICTIONS_FROZEN',seed=seed,method=method,
                    target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,
                    views=list(VIEWS),modes=['direct']):
            raise ValueError(f'Missing frozen full LoRa proof for {name}')
        for v in VIEWS:
            rows=[]
            for rx in TARGET:
                a=np.load(p/f'{rx}__{v}__direct.npy',allow_pickle=False)
                if a.shape!=(LIMIT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=len(CLASSES):
                    raise ValueError('Invalid full LoRa prediction')
                rows.append(a)
            preds[name,v]=np.concatenate(rows)
    return preds


def target_scores(preds,truth,methods=METHODS):
    answer=np.concatenate([truth[r] for r in TARGET])
    by_seed=[]
    for name,seed,m in cases(methods):
        for v in VIEWS:
            p=preds[name,v]
            ok=p==answer
            pl=preds[case(seed,'plain_pl'),v]==answer
            sup=preds[case(seed,'continued_sup'),v]==answer
            cm=np.bincount(len(CLASSES)*answer+p,minlength=len(CLASSES)**2).reshape(len(CLASSES),-1)
            by_seed.append(dict(method=m,seed=seed,view=v,n=len(ok),correct=int(ok.sum()),
                accuracy=float(ok.mean()),macro_f1=macro_f1(answer,p),
                worst_RX=float(ok.reshape(len(TARGET),LIMIT).mean(1).min()),
                by_rx=dict(zip(TARGET,ok.reshape(len(TARGET),LIMIT).mean(1).tolist(),strict=True)),
                by_class=[dict(tx=tx,n=int(cm[i].sum()),correct=int(cm[i,i])) for i,tx in enumerate(CLASSES)],
                gained_PL=int((ok&~pl).sum()),lost_PL=int((~ok&pl).sum()),
                gained_sup=int((ok&~sup).sum()),lost_sup=int((~ok&sup).sum())))
            base=preds.get((case(seed,'adaptive_tri'),v))
            if base is not None:
                ref=base==answer
                by_seed[-1].update(gained_tri=int((ok&~ref).sum()),lost_tri=int((~ok&ref).sum()))
    summary=[]
    for m in methods:
        for which in (*VIEWS,'practical_mean'):
            chosen=[r for r in by_seed if r['method']==m
                    and (r['view']==which or (which=='practical_mean' and r['view']!='clean'))]
            buckets=[[r for r in chosen if r['seed']==s] for s in SEEDS]
            values=[mean(r['accuracy'] for r in x) for x in buckets]
            delta=[mean((r['gained_PL']-r['lost_PL'])/r['n'] for r in x) for x in buckets]
            summary.append(dict(method=m,view=which,mean=mean(values),sample_std=stdev(values),
                  mean_vs_PL=mean(delta),seed_values=values,seed_deltas_PL=delta))
    return dict(status='SCORED_COMPLETE',truth_last=True,by_seed=by_seed,summary=summary)


def verify_control_policies(controls, output):
    for seed in SEEDS:
        name=f'{seed}.json'
        old=read(controls/'policy'/name)
        new=read(output/'policy'/name)
        if (old!=new or old.get('supervised_epochs')!=FULL_EPOCHS+2*EPOCHS
                or old.get('seed')!=seed):
            raise ValueError('Tri set comparisons require the same frozen E140 teacher policy')


def collect(submit,ssh_config,dataset,source,output,controls=None):
    run_root=PurePosixPath(read(submit)['run_root'])
    root=shlex.quote(str(run_root))
    methods=methods_for(run_root)
    names=[n for n,_,_ in cases(methods)]
    sets=run_root.name==SET_RUN
    if sets!=(controls is not None):
        raise ValueError('The frozen E168 controls are required only for the Tri set comparison')
    prior=matrix(controls,WARMUP_METHODS) if controls is not None else {}
    conn=['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    wait_for_predictions(conn,root)
    additions=' '.join(f'F/{n}/epochs.jsonl F/{n}/accepted_U.npy' for n in names)
    if sets:
        additions+=' '+' '.join(f'F/{n}/accepted_sets.npy' for n in names)
    if run_root.name==WARMUP_RUN:
        additions+=' '+' '.join(f'warmup/F/{case(s,"continued_sup")}/epochs.jsonl' for s in SEEDS)
    raw=subprocess.check_output([*conn,f'tar -czf - -C {root} predictions source_audit.json policy {additions}'],
                               timeout=180)
    output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=BytesIO(raw),mode='r:gz') as ar:
        for m in ar:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('predictions','source_audit.json','policy','F','warmup'):
                raise ValueError('Unexpected full LoRa artifact')
            if m.isdir():
                continue
            if not m.isfile() or p.suffix not in ('.json','.jsonl','.npy'):
                raise ValueError('Invalid full LoRa score-side member')
            dest=output.joinpath(*p.parts)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as out:
                out.write(ar.extractfile(m).read())
    if sets:
        verify_control_policies(controls,output)
    predictions={**prior,**matrix(output,methods)}
    truth={}
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as ar:
        for i,rx in enumerate(TARGET):
            with recording(ar,f'{rx}_test.h5',output) as h5:
                truth[rx]=label_ids(h5,LIMIT)[packet_indices(LIMIT,i)]
    write(output/'scores.json',target_scores(predictions,truth,methods))
    score_refined_u(dataset,output,source,names,EPOCHS,sets=sets)
    print('FULL_LORA_SCORED',output/'scores.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('submit','ssh-config','dataset','source','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--controls',type=Path)
    collect(**vars(p.parse_args()))
