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
from .full_dura import (COVERAGE_RUN, MODES, PROJECTED_RUN, SHAPE_RUN, STACK_RUN, STAGED_RUN, SWEEP,
                        SWEEP_RUN, modes_for)
from .full_dura_core import COVERAGE_RECIPES
from .full_dura_detail import (CONFIRM_NEW_SEEDS, DETAIL_CONFIRM_RUN, DETAIL_DOSE_MODES, DETAIL_DOSE_RUN,
                               DETAIL_MODES, DETAIL_RUN, DETAIL_SEEDS)
from .full_dura_r69_lora import (R69_ACTION_MODES, R69_ACTION_RUN, R69_LORA_RUN,
                                 R69_MODES, R69_SEEDS, R69_TRACE_MODES)
from .full_dura_trace_source import TRACE_RUN
from .full_native_joint import ADAPTED_JOINT_RUN, ADAPTED_MODES, JOINT_MODES, JOINT_RUN
from .full_packet_data import LIMIT
from .full_supervised import SEEDS
from .full_tri_repl_lora import ADAPT_METHODS, REPL_METHODS
from .scoring import macro_f1


PAIR_MATRICES=(JOINT_MODES,ADAPTED_MODES)
R69_MATRICES=(R69_MODES,*PAIR_MATRICES,R69_ACTION_MODES,R69_TRACE_MODES)


def matrix(output,modes=MODES,*,seeds=None):
    pred={}
    act={}
    detail=modes in (DETAIL_MODES,DETAIL_DOSE_MODES,*R69_MATRICES)
    seeds=seeds or (DETAIL_SEEDS if detail else SEEDS)
    for s in seeds:
        p=output/'predictions'/str(s)
        receipt=read(p/'complete.json')
        activation=receipt.pop('activation')
        if receipt!=dict(status='PREDICTIONS_FROZEN',seed=s,modes=list(modes),target_truth_read=False,
                      receivers=list(TARGET),views=list(VIEWS),per_receiver=LIMIT):
            raise ValueError('Incomplete matched DURA prediction proofs')
        if set(activation)!={f'{r}/{v}' for r in TARGET for v in VIEWS}:
            raise ValueError('Incomplete per-receiver activation docket')
        for k,row in activation.items():
            if (row['n']!=LIMIT or not 0<=row['active']<=LIMIT or row['IQ_exact']+row['active']!=LIMIT
                    or not 0<=row['mean_dose']<=1.):
                raise ValueError('Invalid per-view dosage bookkeeping')
            if detail and (not row['active']<=row['detail_active']<=LIMIT
                           or row['detail_active']-row['active']!=row['detail_extra']):
                raise ValueError('Invalid per-mode high-coherence bookkeeping')
            if modes in R69_MATRICES and not 0<=row['r69_active']<=LIMIT:
                raise ValueError('Invalid original DURA LoRa gate count')
            if modes==R69_ACTION_MODES and not 0<=row['action_active']<=LIMIT:
                raise ValueError('Invalid LoRa action-corrected gate count')
            if modes in (R69_TRACE_MODES,ADAPTED_MODES) and not 0<=row['trace_active']<=LIMIT:
                raise ValueError('Invalid LoRa trace-trained gate count')
        act[s]=activation
        for v in VIEWS:
            for m in modes:
                values=[]
                for r in TARGET:
                    q=np.load(p/f'{r}__{v}__{m}.npy',allow_pickle=False)
                    if q.shape!=(LIMIT,) or q.dtype.kind not in 'iu' or q.min()<0 or q.max()>=len(CLASSES):
                        raise ValueError('Invalid DURA class predictions')
                    values.append(q)
                pred[s,m,v]=np.concatenate(values)
    return pred,act


def scores(pred,act,truth,modes=MODES,*,seeds=None):
    labels=np.concatenate([truth[r] for r in TARGET])
    n=len(labels)
    rows=[]
    detail=modes in (DETAIL_MODES,DETAIL_DOSE_MODES,*R69_MATRICES)
    seeds=seeds or (DETAIL_SEEDS if detail else SEEDS)
    for s in seeds:
        for v in VIEWS:
            direct=pred[s,'direct',v]==labels
            fixed=pred[s,'fixed',v]==labels
            active=sum(act[s][f'{r}/{v}']['active'] for r in TARGET)
            for mode in modes:
                p=pred[s,mode,v]
                c=p==labels
                cm=np.bincount(10*labels+p,minlength=100).reshape(10,10)
                count=0 if mode=='direct' else active
                method=next((m for m in (*REPL_METHODS,*ADAPT_METHODS) if mode.startswith(f'{m}_')),None)
                if modes in PAIR_MATRICES and method is not None:
                    routed=mode==f'{method}_routed'
                    name='direct' if routed else f'{method}_direct'
                    reference=pred[s,name,v]
                    if name==mode:
                        count=0
                    else:
                        key='trace_active' if modes==ADAPTED_MODES else 'r69_active'
                        count=sum(act[s][f'{r}/{v}'][key] for r in TARGET)
                    if (p!=reference).sum()>count:
                        raise ValueError('Combined LoRa DURA changes exceed the matched Tri gate')
                    right=reference==labels
                    changes=dict(gained_own_direct=int((c&~right).sum()),lost_own_direct=int((~c&right).sum()))
                    if routed:
                        before=pred[s,'r69_trace_half',v]==labels
                        changes.update(gained_trace_tri_set=int((c&~before).sum()),
                                       lost_trace_tri_set=int((~c&before).sum()))
                elif mode in (*R69_ACTION_MODES[4:],*R69_TRACE_MODES[-2:]):
                    calibrated=mode in R69_ACTION_MODES[-2:]
                    trace=mode in R69_TRACE_MODES[-2:]
                    k='trace_active' if trace else 'action_active' if calibrated else 'r69_active'
                    count=sum(act[s][f'{r}/{v}'][k] for r in TARGET)
                    if (p!=pred[s,'direct',v]).sum()>count:
                        raise ValueError('Original DURA LoRa changes exceed the gate')
                    ref=pred[s,'coherent_half',v]==labels
                    changes=dict(gained_half=int((c&~ref).sum()),lost_half=int((~c&ref).sum()))
                    if calibrated or trace:
                        parent=pred[s,'r69_aligned_half' if mode.endswith('_half') else 'r69_aligned',v]==labels
                        changes.update(gained_initial=int((c&~parent).sum()),lost_initial=int((~c&parent).sum()))
                elif detail and mode in DETAIL_DOSE_MODES[3:]:
                    count=sum(act[s][f'{r}/{v}']['detail_active'] for r in TARGET)
                    extra=count-active
                    unchanged=pred[s,'protect_shape',v]==labels
                    if (p!=pred[s,'protect_shape',v]).sum()>extra:
                        raise ValueError('Changes exceed the high-coherence gate')
                    changes=dict(extra_active=extra,gained_protected=int((c&~unchanged).sum()),
                                 lost_protected=int((~c&unchanged).sum()))
                else:
                    changes={}
                rows.append(dict(seed=s,mode=mode,view=v,n=n,accuracy=float(c.mean()),macro_f1=macro_f1(labels,p),
                   changed_vs_direct=int((p!=pred[s,'direct',v]).sum()),
                   gained_direct=int((c&~direct).sum()),lost_direct=int((~c&direct).sum()),
                   gained_fixed=int((c&~fixed).sum()),lost_fixed=int((~c&fixed).sum()),
                   by_class={str(k):[int(cm[i,i]),int(cm[i].sum())] for i,k in enumerate(CLASSES)},
                   by_rx=dict(zip(TARGET,c.reshape(len(TARGET),LIMIT).mean(1).tolist(),strict=True)),
                   active=count,IQ_exact=n-count,**changes))
    summary={}
    for m in modes:
        by={}
        for v in (*VIEWS,'practical_mean'):
            table={}
            for s in seeds:
                q=[x for x in rows if x['mode']==m and x['seed']==s
                   and (x['view']==v or (v=='practical_mean' and x['view']!='clean'))]
                table[s]=dict(accuracy=statistics.mean(x['accuracy'] for x in q),
                    delta=statistics.mean((x['gained_direct']-x['lost_direct'])/x['n'] for x in q),
                    vs_fixed=statistics.mean((x['gained_fixed']-x['lost_fixed'])/x['n'] for x in q))
            by[v]=dict(mean=statistics.mean(x['accuracy'] for x in table.values()),
                sample_std=statistics.stdev(x['accuracy'] for x in table.values()) if len(seeds)>1 else None,
                seed=table)
        summary[m]=by
    return dict(status='SCORED_COMPLETE',truth_last=True,by_seed=rows,summary=summary)


def collect(submit,ssh_config,dataset,output,sweep=False):
    run=PurePosixPath(read(submit)['run_root'])
    root=shlex.quote(str(run))
    frozen=run.name in (SHAPE_RUN,SWEEP_RUN,STACK_RUN) or sweep
    cmd=['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    wait_for_predictions(cmd,root)
    if run.name in (JOINT_RUN,ADAPTED_JOINT_RUN):
        extra=''
    elif run.name in (R69_LORA_RUN,R69_ACTION_RUN,TRACE_RUN):
        extra=' '+' '.join(f'G/{stage}/{s}/source_support.json G/{stage}/{s}/steps.jsonl'
                           for stage in ('teacher','student') for s in R69_SEEDS)
        extra+=' '+' '.join(f'G/gate/{s}/training.json' for s in R69_SEEDS)
        if run.name==TRACE_RUN:
            extra+=' source_trace/complete.json'
    elif run.name==DETAIL_CONFIRM_RUN:
        extra=' '+' '.join(f'G/{s}/source_support.json G/{s}/steps.jsonl' for s in CONFIRM_NEW_SEEDS)
    elif run.name==DETAIL_DOSE_RUN:
        extra=''
    elif run.name==DETAIL_RUN:
        extra=' '+' '.join(f'G/{s}/source_support.json G/{s}/steps.jsonl' for s in DETAIL_SEEDS)
    elif run.name==COVERAGE_RUN:
        extra=' '+' '.join(f'G/{mode}/{s}/source_support.json G/{mode}/{s}/steps.jsonl'
                           for mode in COVERAGE_RECIPES for s in SEEDS)
    elif run.name==PROJECTED_RUN:
        extra=' '+' '.join(f'G/{s}/source_support.json G/{s}/steps.jsonl' for s in SEEDS)
    elif run.name==STAGED_RUN:
        extra=' '+' '.join(f'G/{stage}/{s}/source_support.json G/{stage}/{s}/steps.jsonl'
                           for stage in ('teacher','student') for s in SEEDS)
    elif frozen:
        extra=''
    else:
        extra=' G/gate/training.json '+' '.join(f'G/{s}/dose_training.json G/{s}/steps.jsonl' for s in SEEDS)
    raw=subprocess.check_output([*cmd,f'tar -czf - -C {root} predictions source_wave{extra}'],
                                timeout=180)
    output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(fileobj=BytesIO(raw),mode='r:gz') as ar:
        for member in ar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] not in ('predictions','G','source_wave','source_trace'):
                raise ValueError('Unexpected DURA score artifact')
            if member.isdir():
                continue
            if not member.isfile() or path.suffix not in ('.npy','.json','.jsonl'):
                raise ValueError('Unexpected DURA score member')
            out=output.joinpath(*path.parts)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('xb') as f:
                f.write(ar.extractfile(member).read())
    modes=tuple(SWEEP) if sweep else modes_for(run)
    seeds=SEEDS if run.name==DETAIL_CONFIRM_RUN else None
    pred,act=matrix(output,modes,seeds=seeds)
    truth={}
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as ar:
        for i,n in enumerate(TARGET):
            with recording(ar,f'{n}_test.h5',output) as h5:
                truth[n]=label_ids(h5,LIMIT)[packet_indices(LIMIT,i)]
    write(output/'scores.json',scores(pred,act,truth,modes,seeds=seeds))
    print('FULL_DURA_SCORED',output/'scores.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('submit','ssh-config','dataset','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--sweep',action='store_true')
    collect(**vars(p.parse_args()))
