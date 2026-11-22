"""Independent truth-last score of three LoRa G seeds and their fixed F controls."""
import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import shlex
import statistics
import subprocess
import tarfile
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read,write
from .collect import wait_for_predictions
from .data import CLASSES,TARGET,VIEWS,label_ids,packet_indices,recording
from .full_dura import MODES
from .full_packet_data import LIMIT
from .full_supervised import SEEDS
from .scoring import macro_f1


def matrix(output):
    pred={}
    act={}
    for s in SEEDS:
        p=output/'predictions'/str(s)
        receipt=read(p/'complete.json')
        activation=receipt.pop('activation')
        if receipt!=dict(status='PREDICTIONS_FROZEN',seed=s,modes=list(MODES),target_truth_read=False,
                      receivers=list(TARGET),views=list(VIEWS),per_receiver=LIMIT):
            raise ValueError('Incomplete matched DURA prediction proofs')
        if set(activation)!={f'{r}/{v}' for r in TARGET for v in VIEWS}:
            raise ValueError('Incomplete per-receiver activation docket')
        for k,row in activation.items():
            if (row['n']!=LIMIT or not 0<=row['active']<=LIMIT or row['IQ_exact']+row['active']!=LIMIT
                    or not 0<=row['mean_dose']<=1.):
                raise ValueError('Invalid per-view dosage bookkeeping')
        act[s]=activation
        for v in VIEWS:
            for m in MODES:
                values=[]
                for r in TARGET:
                    q=np.load(p/f'{r}__{v}__{m}.npy',allow_pickle=False)
                    if q.shape!=(LIMIT,) or q.dtype.kind not in 'iu' or q.min()<0 or q.max()>=len(CLASSES):
                        raise ValueError('Invalid DURA class predictions')
                    values.append(q)
                pred[s,m,v]=np.concatenate(values)
    return pred,act


def scores(pred,act,truth):
    labels=np.concatenate([truth[r] for r in TARGET])
    n=len(labels)
    rows=[]
    for s in SEEDS:
        for v in VIEWS:
            direct=pred[s,'direct',v]==labels
            fixed=pred[s,'fixed',v]==labels
            active=sum(act[s][f'{r}/{v}']['active'] for r in TARGET)
            for mode in MODES:
                p=pred[s,mode,v]
                c=p==labels
                cm=np.bincount(10*labels+p,minlength=100).reshape(10,10)
                rows.append(dict(seed=s,mode=mode,view=v,n=n,accuracy=float(c.mean()),macro_f1=macro_f1(labels,p),
                   changed_vs_direct=int((p!=pred[s,'direct',v]).sum()),
                   gained_direct=int((c&~direct).sum()),lost_direct=int((~c&direct).sum()),
                   gained_fixed=int((c&~fixed).sum()),lost_fixed=int((~c&fixed).sum()),
                   by_class={str(k):[int(cm[i,i]),int(cm[i].sum())] for i,k in enumerate(CLASSES)},
                   by_rx=dict(zip(TARGET,c.reshape(len(TARGET),LIMIT).mean(1).tolist(),strict=True)),
                   active=0 if mode=='direct' else active,IQ_exact=n if mode=='direct' else n-active))
    summary={}
    for m in MODES:
        by={}
        for v in (*VIEWS,'practical_mean'):
            table={}
            for s in SEEDS:
                q=[x for x in rows if x['mode']==m and x['seed']==s
                   and (x['view']==v or (v=='practical_mean' and x['view']!='clean'))]
                table[s]=dict(accuracy=statistics.mean(x['accuracy'] for x in q),
                    delta=statistics.mean((x['gained_direct']-x['lost_direct'])/x['n'] for x in q),
                    vs_fixed=statistics.mean((x['gained_fixed']-x['lost_fixed'])/x['n'] for x in q))
            by[v]=dict(mean=statistics.mean(x['accuracy'] for x in table.values()),
                sample_std=statistics.stdev(x['accuracy'] for x in table.values()),seed=table)
        summary[m]=by
    return dict(status='SCORED_COMPLETE',truth_last=True,by_seed=rows,summary=summary)


def collect(submit,ssh_config,dataset,output):
    root=shlex.quote(read(submit)['run_root'])
    cmd=['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    wait_for_predictions(cmd,root)
    traces=' '.join(f'G/{s}/dose_training.json G/{s}/steps.jsonl' for s in SEEDS)
    raw=subprocess.check_output([*cmd,f'tar -czf - -C {root} predictions source_wave G/gate/training.json {traces}'],
                                timeout=180)
    output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=BytesIO(raw),mode='r:gz') as ar:
        for member in ar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('predictions','G','source_wave'):
                raise ValueError('Unexpected DURA score artifact')
            if member.isdir():
                continue
            if not member.isfile() or path.suffix not in ('.npy','.json','.jsonl'):
                raise ValueError('Unexpected DURA score member')
            out=output.joinpath(*path.parts)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('xb') as f:
                f.write(ar.extractfile(member).read())
    pred,act=matrix(output)
    truth={}
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as ar:
        for i,n in enumerate(TARGET):
            with recording(ar,f'{n}_test.h5',output) as h5:
                truth[n]=label_ids(h5,LIMIT)[packet_indices(LIMIT,i)]
    write(output/'scores.json',scores(pred,act,truth))
    print('FULL_DURA_SCORED',output/'scores.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('submit','ssh-config','dataset','output'):
        p.add_argument('--'+k,type=Path,required=True)
    collect(**vars(p.parse_args()))
