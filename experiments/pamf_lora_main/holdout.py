"""Receiver-held-out supervised LoRa teachers and source-only calibration."""
import argparse
from functools import partial
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import seed_all
from .data import SOURCE, VIEWS
from .packet_models import SEED, packet_tri
from .packet_runner import schedule
from .refine_runner import source_handles, start
from .repair_features import CoherentPacket
from .runner import activate
from .supervised_extension import lr_scale

FOLDS = (2, 3, 4)
EPOCHS = 180
STEPS = 40


def labeled(source, cache, role, held):
    """Never expose held-receiver L or other-V IQ to the fold worker."""
    if role not in ('L_s', 'V'):
        raise ValueError('Expected a permitted labeled source role')
    c = read(source/'contract.json')
    with np.load(source/f'{role}_metadata.npz', allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids'][role]:
            raise ValueError('Expected the exact permitted labeled source role')
        keep = m['rx_i']!=held if role=='L_s' else m['rx_i']==held
        y, rx, ids = [m[k][keep] for k in ('y', 'rx_i', 'ids')]
    x = np.load(cache/f'{role}.npy', mmap_mode='r', allow_pickle=False)
    if x.shape!=(len(VIEWS),c['counts'][role],6,2,256):
        raise ValueError('Held-receiver source cache shape differs')
    return np.ascontiguousarray(x[:,keep]), y, rx, ids


def batches(y, rx, rng):
    receivers=np.unique(rx)
    if len(receivers)!=4:
        raise ValueError('Fold train requires four receivers')
    cells=[[rng.permutation(np.flatnonzero((y==c)&(rx==r))) for r in receivers] for c in range(10)]
    if any(len(a)!=32 for row in cells for a in row):
        raise ValueError('Every fold TX/receiver cell must have 32 physical packets')
    for first in range(0,32,4):
        groups=rng.permutation(10).reshape(5,2)
        cursor=np.full(10,first,dtype=np.int64)
        for omitted in range(5):
            chosen=groups[np.arange(5)!=omitted].ravel()
            yield np.asarray([r[cursor[c]] for c in chosen for r in cells[c]],dtype=np.int64)
            cursor[chosen]+=1


def construct(device):
    return CoherentPacket(packet_tri('packet_sup',device)).to(device)


@torch.no_grad()
def measure(model, x, y):
    model.eval()
    logits = torch.cat([model(z)[0] for z in x.split(48)])
    pred = logits.argmax(1)
    good = pred == y
    return dict(correct=int(good.sum()), total=len(y), ce=float(F.cross_entropy(logits,y)),
                by_class=torch.bincount(y[good],minlength=10).tolist())


def train(rt, held):
    source,_,cache,c = source_handles(rt)
    if held not in FOLDS:
        raise ValueError('Unregistered held receiver')
    folder=Path(rt['run_root'])/'F'/f'rx{held}'
    folder.mkdir(parents=True,exist_ok=False)
    seed_all(SEED)
    dev='cuda:0'
    v,yn,rn,ids = labeled(source,cache,'L_s',held)
    x,y=tensor(v,dev),tensor(yn,dev)
    model=construct(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=5e-4)
    # Stretch the verified two-stage supervised schedule to a cold start.
    lr=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:lr_scale(s*6000//(EPOCHS*STEPS)))
    label_rng=np.random.default_rng(721)
    scene_rng=np.random.default_rng(267)
    step=0
    with (folder/'epochs.jsonl').open('x',encoding='utf8') as trace:
        for epoch in range(1,EPOCHS+1):
            alpha=min(.35,max(0.,(epoch-54)*.035))
            seen=0
            for b in batches(yn,rn,label_rng):
                i=tensor(b,dev)
                scene=tensor(scene_rng.integers(1,len(VIEWS),size=len(b)),dev)
                model.train()
                if alpha:
                    clean,noisy=model(torch.cat((x[0,i],x[scene,i])))[0].chunk(2)
                    loss=F.cross_entropy(clean,y[i])+alpha*F.cross_entropy(noisy,y[i])
                else:
                    loss=F.cross_entropy(model(x[0,i])[0],y[i])
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step();lr.step()
                seen+=len(b)
                step+=1
            if seen!=len(ids) or step!=STEPS*epoch:
                raise ValueError('Fold coverage or update budget differs')
            if epoch in (1,54,EPOCHS) or epoch%12==0:
                row=dict(epoch=epoch,step=step,lr=opt.param_groups[0]['lr'],L=measure(model,x[0],y))
                trace.write(json.dumps(row,allow_nan=False)+'\n');trace.flush()
                print('HOLDOUT_TRAIN',held,json.dumps(row,allow_nan=False),flush=True)
    torch.save(model.state_dict(),folder/'model.pt')
    write(folder/'complete.json',dict(status='SOURCE_FROZEN',held=held,seed=SEED,steps=step,
          source_contract=c,train_ids=ids.tolist(),cold_start=True,V_reads=0,U_reads=0,target_reads=0))


@torch.no_grad()
def view_probability(model,x):
    return [model(v)[0].softmax(1) for v in (x,x[:,::2].repeat(1,2,1,1),x[:,1::2].repeat(1,2,1,1))]


def view_evidence(model,x):
    """Three estimates from the packet and its disjoint chirp subsets."""
    return probability_evidence(view_probability(model,x))


def probability_evidence(probability):
    all_prob,odd,even=probability
    c,p=all_prob.max(1)
    a,i=odd.max(1)
    b,j=even.max(1)
    agree=(p==i)&(p==j)
    return c,p,torch.minimum(c,torch.minimum(a,b)),agree


def summary(c,p,g,agree,y):
    out={}
    for t in (.75,.85,.90,.95):
        out[str(t)]={}
        for name,mask in (('single',c>=t),('three_views',agree&(g>=t))):
            out[str(t)][name]=dict(n=int(mask.sum()),correct=int((mask&(p==y)).sum()))
    return out


def calibrate(p, confidence, agree, y, rx):
    # Split inside each TX/RX cell. The second half checks the one threshold
    # chosen on the first half, without retuning classes that fail.
    take=torch.zeros(len(y),dtype=torch.bool,device=y.device)
    for k in range(10):
        for r in FOLDS:
            at=torch.where((y==k)&(rx==r))[0]
            take[at[::2]]=True
    rules=[]
    for cls in range(10):
        for threshold in (.85,.90,.95):
            elig=agree&(p==cls)&(confidence>=threshold)
            evidence=[]
            for part in (take,~take):
                selected=part&elig
                hits=selected&(y==cls)
                n=int(selected.sum());h=int(hits.sum())
                support=int((torch.bincount(rx[hits],minlength=5)>=2).sum())
                lower=0.
                if n:
                    prop=h/n; z=1.96
                    lower=(prop+z*z/(2*n)-z*math.sqrt((prop*(1-prop)+z*z/(4*n))/n))/(1+z*z/n)
                evidence.append(dict(n=n,hits=h,support=support,wilson=lower))
            a,b=evidence
            qualified=lambda q:q['n']>=8 and q['hits']/q['n']>=.875 and q['support']>=2 and q['wilson']>=.55
            if qualified(a):
                rules.append(dict(class_id=cls,threshold=threshold,qualified=qualified(b),cal=a,check=b))
                break
    return rules


def audit(rt):
    source,_,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    rows={}; parts=[]
    for held in FOLDS:
        folder=root/'F'/f'rx{held}'
        m=read(folder/'complete.json')
        if not (m['status']=='SOURCE_FROZEN' and m['held']==held and m['cold_start']
                and m['V_reads']==m['U_reads']==m['target_reads']==0 and m['seed']==SEED
                and m['steps']==EPOCHS*STEPS and m['source_contract']==c and len(m['train_ids'])==1280
                and all(not k.startswith(SOURCE[held]+'/') for k in m['train_ids'])):
            raise ValueError('Illegal fold ancestry or incomplete checkpoint')
        model=construct('cuda:0')
        model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda:0',weights_only=True),strict=True)
        model.eval()
        a,lab,rx,ids=labeled(source,cache,'V',held)
        if set(ids)&set(m['train_ids']) or len(ids)!=160:
            raise ValueError('Held source scoring rows differ from the physical contract')
        x,y,rr=tensor(a,'cuda:0'),tensor(lab,'cuda:0'),tensor(rx,'cuda:0')
        cf,p,g,agree=(torch.cat(z) for z in zip(*(view_evidence(model,b) for b in x[0].split(48)),strict=True))
        parts.append((cf,p,g,agree,y,rr))
        rows[SOURCE[held]]=dict(held=held,views={v:measure(model,x[i],y) for i,v in enumerate(VIEWS)},
            confidence=summary(cf,p,g,agree,y))
    cf,p,g,agree,y,rr=(torch.cat(x) for x in zip(*parts,strict=True))
    rules=calibrate(p,g,agree,y,rr)
    write(root/'holdout_audit.json',dict(status='SOURCE_HOLDOUT_FROZEN',rows=rows,rules=rules,
          confidence=summary(cf,p,g,agree,y),F_folds=list(FOLDS),seed=SEED,
          source_contract=c,U_reads=0,target_reads=0))


def smoke(rt):
    source,_,cache,_=source_handles(rt)
    v,y,rx,_=labeled(source,cache,'L_s',FOLDS[0])
    b=np.asarray(list(batches(y,rx,np.random.default_rng(721))))
    assert b.shape==(STEPS,32) and np.array_equal(np.sort(b.ravel()),np.arange(len(y)))
    assert not (rx==FOLDS[0]).any()
    model=construct('cpu')
    r,fx=model(tensor(np.concatenate((v[0,:3],v[1,:3]))))
    F.cross_entropy(r,tensor(np.tile(y[:3],2))).backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    assert r.shape==(6,10) and float(norm)>0 and bool(torch.isfinite(fx).all())
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',train_receiver_excluded=FOLDS[0],
          n=len(y),gradient=float(norm),V_reads=0,U_reads=0,target_reads=0))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch=partial(start,module=__spec__.name)
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        if launch(rt,'smoke','holdout').wait():
            raise RuntimeError('Held-receiver source smoke failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(rt,[('train',str(r)) for r in FOLDS],launch)
        write(state,dict(status='SOURCE_AUDIT'),exclusive=False)
        schedule(rt,[('audit','all')],launch)
        write(state,dict(status='SOURCE_ONLY_COMPLETE',U_reads=0,target_reads=0),exclusive=False)
    except Exception as err:
        write(state,dict(status='FAILED',error=str(err)),exclusive=False)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=('dispatch','smoke','train','audit'))
    parser.add_argument('runtime',type=Path)
    parser.add_argument('name',nargs='?')
    a=parser.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if a.kind=='train':
            train(rt,int(a.name))
        elif a.kind=='audit':
            audit(rt)
        else:
            smoke(rt)
