"""Run the frozen-source LoRa attribution matrix and then truth-blind inference."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from time import sleep

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import TARGET, VIEWS
from .packet_data import TARGET_LIMIT
from .packet_models import SEED
from .packet_runner import schedule
from .refine_f import EPOCHS, METHODS, PARENT, build, class_policy, measure, source_contract, train
from .refine_g import ActiveDura, G_EPOCHS, METHODS_G, load_g_parent
from .refine_g import train as train_g
from .runner import activate


RUN='20261122-phase1-lora-packet-refine-m6-r01'
READOUTS={m: ('direct','parent_dura',*METHODS_G) if m=='spectral_tri' else ('direct',) for m in METHODS}


def select_gpu(exclude=()):
    query = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits']
    for _ in range(30):
        rows = (tuple(map(int, line.split(','))) for line in subprocess.check_output(query, text=True).splitlines())
        available = [(load, -free, gpu) for gpu, load, free in rows if free >= 12_288 and gpu not in exclude]
        if available:
            return min(available)[-1]
        sleep(20)
    raise RuntimeError('No GPU has 12 GiB free')


def source_handles(rt):
    source, parent = Path(rt['source_root']), Path(rt['parent_root'])
    cache=parent/'source_views'
    c=source_contract(source)
    v=read(cache/'complete.json')
    if (parent.name!=PARENT or v['status']!='SOURCE_READY' or v['target_read']
            or v['input_fs']!=1_000_000 or v['counts']!=c['counts'] or v['views']!=list(VIEWS)):
        raise ValueError('Aligned 1 MHz source cache from the exact parent is required')
    return source,parent,cache,c


def smoke(rt):
    source,parent,cache,c=source_handles(rt)
    device='cpu'
    x=tensor(np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)[0])
    with np.load(source/'V_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids']['V']:
            raise ValueError('Source validation order differs')
        y,rx=tensor(m['y']),tensor(m['rx_i'])
    raw=build(parent,c,'raw_pl',device).eval()
    spec=build(parent,c,'spectral_tri',device).eval()
    quality,conf,pred=measure(spec,x,y)
    policy=class_policy(conf,pred,y,rx)
    g=load_g_parent(parent,c,device,ActiveDura()).eval()
    with torch.no_grad():
        equal=torch.equal(raw(x[:8])[0],spec(x[:8])[0])
        out,gate=g(x[:24],with_gate=True)
        if (not equal or not torch.isfinite(out).all()
                or not torch.equal(out[gate==0],x[:24][gate==0])):
            raise ValueError('Actual source checkpoint logit or restoration identity smoke failed')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',V_clean=quality,
          class_policy=policy,initial_logits_equal=equal,target_reads=0,F_read_by_G=0))


def load_f(root,c,name,device):
    path=root/'F'/name
    if read(path/'complete.json')!=dict(status='SOURCE_FROZEN',method=name,seed=SEED,epochs=EPOCHS,
                                       source_contract=c,target_read=False):
        raise ValueError('Refined F source freeze differs')
    f=build(None,c,name,device,load_parent=False).eval()
    f.load_state_dict(torch.load(path/'model.pt',map_location=device,weights_only=True),strict=True)
    return f


def load_restorers(root,parent,c,device):
    gs={'parent_dura':load_g_parent(parent,c,device).eval()}
    for mode in METHODS_G:
        path=root/'G'/mode
        if read(path/'complete.json')!=dict(status='SOURCE_FROZEN',mode=mode,seed=SEED,epochs=G_EPOCHS,
                                          source_contract=c,F_reads=False,TX_labels_read=False,target_read=False):
            raise ValueError('Refined G source freeze differs')
        g=ActiveDura().to(device).eval()
        g.load_state_dict(torch.load(path/'model.pt',map_location=device,weights_only=True),strict=True)
        gs[mode]=g
    return gs


@torch.no_grad()
def infer(f,gs,x):
    labels={k:[] for k in ('direct',*gs)}
    gate={k:[] for k in gs}
    for batch in x.split(24):
        scores={'direct':f(batch)[0]}
        for k,g in gs.items():
            restored,amount=g(batch,with_gate=True)
            if not torch.equal(restored[amount==0],batch[amount==0]):
                raise ValueError('DURA bypass changed an observation')
            scores[k]=f(restored)[0]
            gate[k].extend(amount.tolist())
        for k,s in scores.items():
            if not torch.isfinite(s).all():
                raise ValueError('Refined packet inference is nonfinite')
            labels[k].extend(s.argmax(1).tolist())
    return {k:np.asarray(v,dtype=np.int16) for k,v in labels.items()},gate


def source_audit(root,source,parent,cache,c):
    dev='cuda:0'
    x=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    with np.load(source/'V_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids']['V']:
            raise ValueError('Source audit physical rows differ')
        y,rx=m['y'],m['rx_i']
    gs=load_restorers(root,parent,c,dev)
    detail={}
    for name in METHODS:
        f=load_f(root,c,name,dev)
        detail[name]={}
        for i,v in enumerate(VIEWS):
            pred,gate=infer(f,gs if name=='spectral_tri' else {},tensor(x[i],dev))
            base=pred['direct']==y
            detail[name][v]={k:dict(correct=int((p==y).sum()),total=len(y),
                   by_rx=[float((p[rx==i]==y[rx==i]).mean()) for i in range(5)],
                   by_class=[int(((p==j)&(y==j)).sum()) for j in range(10)],
                   gained=int((~base&(p==y)).sum()),lost=int((base&(p!=y)).sum()),
                   active=sum(a>0 for a in gate.get(k,()))) for k,p in pred.items()}
    write(root/'source_audit.json',dict(status='SOURCE_AUDIT_COMPLETE',target_reads=0,records=detail))


def predict_models(root,parent,target,name,model,restore,modes):
    if read(parent/'state.json')['status']!='PREDICTIONS_FROZEN':
        raise ValueError('Incomplete immutable parent packet view source')
    if target!=parent/'target_views' or read(target/'complete.json')!=dict(status='TARGET_READY',
             target_truth_read=False,receivers=list(TARGET),per_receiver=TARGET_LIMIT,views=list(VIEWS)):
        raise ValueError('Exact same blinded target rows and seven views required')
    dev='cuda:0'
    out=root/'predictions'/name
    out.mkdir(parents=True,exist_ok=False)
    counts={}
    for rx in TARGET:
        x=np.load(target/f'{rx}.npy',mmap_mode='r',allow_pickle=False)
        if x.shape!=(len(VIEWS),TARGET_LIMIT,6,2,256):
            raise ValueError('Target view shape differs')
        for i,v in enumerate(VIEWS):
            pred,gate=infer(model,restore,tensor(x[i],dev))
            for k,p in pred.items():
                np.save(out/f'{rx}__{v}__{k}.npy',p)
            counts[f'{rx}/{v}']={k:sum(a==0 for a in g) for k,g in gate.items()}
    write(out/'complete.json',dict(status='PREDICTIONS_FROZEN',method=name,seed=SEED,target_truth_read=False,
        receivers=list(TARGET),per_receiver=TARGET_LIMIT,views=list(VIEWS),modes=list(modes),
        pass_through=counts))


def predict(root,parent,c,target,name):
    dev='cuda:0'
    f=load_f(root,c,name,dev)
    gs=load_restorers(root,parent,c,dev) if name=='spectral_tri' else {}
    return predict_models(root,parent,target,name,f,gs,READOUTS[name])


def worker(rt,kind,name):
    activate(rt)
    source,parent,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    if kind=='smoke':
        return smoke(rt)
    if kind=='train_f':
        return train(source,cache,parent,root/'F'/name,name)
    if kind=='train_g':
        return train_g(source,cache,parent,root/'G'/name,name)
    if kind=='source_audit':
        return source_audit(root,source,parent,cache,c)
    if kind=='predict':
        return predict(root,parent,c,Path(rt['target_views']),name)
    raise ValueError(f'Unknown LoRa refinement job {kind}')


def start(rt,kind,name,gpu=None,*,module=None,extra_keys=()):
    root=Path(rt['run_root'])
    cfg=root/'configs'/f'{kind}-{name}.json'
    keys=['run_root','source_root','parent_root','frozen_sources']
    if kind=='predict':
        keys.append('target_views')
    write(cfg,{k:rt[k] for k in (*keys,*extra_keys)})
    cmd=[sys.executable,'-u','-m',module or __spec__.name,kind,str(cfg),name]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    path=root/'logs'/f'{kind}-{name}.log'
    path.parent.mkdir(exist_ok=True)
    with path.open('x',encoding='utf8') as log:
        p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    print('REFINE_LAUNCHED',kind,name,p.pid,gpu,flush=True)
    return p


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        if start(rt,'smoke','refinement').wait():
            raise RuntimeError('Source no-query refinement smoke failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(rt,[('train_f',m) for m in METHODS]+[('train_g',m) for m in METHODS_G],start)
        write(state,dict(status='SOURCE_AUDIT'),exclusive=False)
        schedule(rt,[('source_audit','all')],start)
        write(state,dict(status='PREDICTING'),exclusive=False)
        schedule(rt,[('predict',m) for m in METHODS],start)
        write(state,dict(status='PREDICTIONS_FROZEN',target_truth_read=False),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind')
    p.add_argument('runtime',type=Path)
    p.add_argument('name',nargs='?',default='')
    a=p.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        worker(rt,a.kind,a.name)
