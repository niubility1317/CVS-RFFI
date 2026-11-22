"""Cold supervised teachers on the complete LoRa source contract."""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import seed_all
from .data import SOURCE, VIEWS
from .full_packet_data import COUNTS, PROTOCOL
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract
from .holdout import FOLDS, calibrate, construct, measure, summary, view_evidence
from .packet_models import SEED
from .packet_runner import schedule
from .refine_f import labeled_weights
from .refine_runner import start
from .runner import activate
from .supervised_extension import lr_scale, metrics

RUN='20261122-phase1-lora-full-supervised-m6-r01'
SEEDS=(2026092701,2026092702,2026092703)
FULL_EPOCHS=84
FOLD_EPOCHS=104
CHECK_EVERY=14


def batches(y,rx,rng):
    """Exact 56-packet cell coverage on four or five receivers."""
    receivers=np.unique(rx)
    nr=len(receivers)
    if nr not in (4,5):
        raise ValueError('Use four held-fold receivers or all five receivers')
    cells=[[rng.permutation(np.flatnonzero((y==c)&(rx==r))) for r in receivers] for c in range(10)]
    if any(len(a)!=56 for row in cells for a in row):
        raise ValueError('Expected the official 56-device-per-receiver labeled cells')
    # 70 class visits with four of five receivers or 56 with all four.
    stream=np.tile(rng.permutation(10),56*nr//4)
    dropped=np.stack([rng.permutation(nr) for _ in range(10)])
    cursor=np.zeros((10,nr),dtype=np.int64)
    turns=np.zeros(10,dtype=np.int64)
    for start_at in range(0,len(stream),8):
        result=[]
        for c in stream[start_at:start_at+8]:
            skip=dropped[c,turns[c]%nr] if nr==5 else -1
            turns[c]+=1
            for r in range(nr):
                if r!=skip:
                    result.append(cells[c][r][cursor[c,r]])
                    cursor[c,r]+=1
        yield np.asarray(result,dtype=np.int64)


def handles(rt):
    source,data=Path(rt['source_root']),Path(rt['parent_root'])
    if data.name!=DATA_RUN:
        raise ValueError('Expected immutable registered full alignment run')
    c=source_contract(source)
    views=data/'source_views'
    state=read(views/'complete.json')
    if state!=dict(status='SOURCE_READY',protocol=PROTOCOL,counts=COUNTS,input_fs=1000000,
                   views=list(VIEWS),U_extra_practical=True,target_read=False):
        raise ValueError('Full source views did not pass the registered data contract')
    return source,views,c


def role(source,views,c,key,device,*,held=None):
    if key not in ('L_s','V'):
        raise ValueError('Supervised worker may only open true L and source V')
    with np.load(source/f'{key}_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids'][key]:
            raise ValueError('Registered full-source physical order differs')
        y,rx,ids=[m[k] for k in ('y','rx_i','ids')]
    x=np.load(views/f'{key}.npy',allow_pickle=False,mmap_mode='r')
    if x.shape!=(7,COUNTS[key],6,2,256) or x.dtype!=np.float32:
        raise ValueError('Full source view shape differs')
    if held is not None:
        take=rx!=held if key=='L_s' else rx==held
        x,y,rx,ids=x[:,take],y[take],rx[take],ids[take]
    return tensor(x,device),y,rx,ids


def row_config(name):
    if name.startswith('seed'):
        seed=int(name[4:])
        if seed not in SEEDS:
            raise ValueError('Unregistered supervised seed')
        return seed,None,FULL_EPOCHS,88
    held=int(name.removeprefix('rx'))
    if held not in FOLDS:
        raise ValueError('Unregistered full LoRa holdout receiver')
    return SEED,held,FOLD_EPOCHS,70


def train(rt,name):
    seed,held,epochs,steps=row_config(name)
    source,views,c=handles(rt)
    root=Path(rt['run_root'])/'F'/name
    root.mkdir(parents=True,exist_ok=False)
    dev='cuda:0'
    seed_all(seed)
    x,yn,rn,ids=role(source,views,c,'L_s',dev,held=held)
    y=tensor(yn,dev)
    if held is None:
        data,vy,vr,_=role(source,views,c,'V',dev)
        v,vy,vr=data[0],tensor(vy,dev),tensor(vr,dev)
    f=construct(dev)
    opt=torch.optim.AdamW(f.parameters(),lr=3e-4,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:lr_scale(s*6000//(epochs*steps)))
    label_rng=np.random.default_rng([721,seed])
    scene_rng=np.random.default_rng([267,seed])
    warm=round(.30*epochs)
    w=torch.ones(10,device=dev)
    t0=time.monotonic()
    step=0
    with (root/'epochs.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(1,epochs+1):
            alpha=min(.35,max(0.,(epoch-warm)*.035))
            if epoch==warm+1:
                f.eval()
                with torch.no_grad():
                    p=torch.cat([f(b)[0].argmax(1) for b in x[0].split(48)])
                _,w=labeled_weights(p,y,True)
            c_loss=torch.zeros((),device=dev)
            p_loss=torch.zeros((),device=dev)
            seen=0
            for b in batches(yn,rn,label_rng):
                i=tensor(b,dev); j=tensor(scene_rng.integers(1,7,size=len(b)),dev)
                f.train()
                if alpha:
                    cl,pr=f(torch.cat((x[0,i],x[j,i])))[0].chunk(2)
                    ce_p=F.cross_entropy(pr,y[i],weight=w)
                else:
                    cl=f(x[0,i])[0]
                    ce_p=c_loss.new_zeros(())
                ce_c=F.cross_entropy(cl,y[i],weight=w)
                opt.zero_grad(set_to_none=True)
                (ce_c+alpha*ce_p).backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True)
                opt.step()
                sch.step()
                c_loss+=ce_c.detach()*len(b)
                p_loss+=ce_p.detach()*len(b)
                seen+=len(b)
                step+=1
            if seen!=len(ids) or step!=epoch*steps:
                raise ValueError('Full supervised source coverage differs')
            if epoch in (1,warm,epochs) or epoch%CHECK_EVERY==0:
                row=dict(epoch=epoch,step=step,seconds=time.monotonic()-t0,
                         L=measure(f,x[0],y),clean_ce=float(c_loss/seen),practical_ce=float(p_loss/seen))
                if held is None:
                    row['V']=metrics(f,v,vy,vr)
                line=json.dumps(row,allow_nan=False)
                log.write(line+'\n')
                log.flush()
                print('FULL_SUPERVISED',name,line,flush=True)
    torch.save(f.state_dict(),root/'model.pt')
    write(root/'complete.json',dict(status='SOURCE_FROZEN',cold_start=True,seed=seed,held=held,
              epochs=epochs,steps=steps*epochs,U_reads=0,target_reads=0,
              V_reads_during_fit=held is None,source_contract=c,train_ids=ids.tolist()))


def load_f(root,c,name,device):
    seed,held,epochs,steps=row_config(name)
    folder=root/'F'/name
    m=read(folder/'complete.json')
    ids=c['role_ids']['L_s']
    if held is not None:
        ids=[id for id in ids if not id.startswith(SOURCE[held]+'/')]
    if (m['status']!='SOURCE_FROZEN' or not m['cold_start']
            or m['seed']!=seed or m['held']!=held or m['epochs']!=epochs or m['steps']!=steps*epochs
            or m['source_contract']!=c or m['U_reads'] or m['target_reads']
            or m['V_reads_during_fit']!=(held is None) or m['train_ids']!=ids):
        raise ValueError(f'Full supervised checkpoint ancestry differs for {name}')
    f=construct(device).eval()
    f.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True),strict=True)
    return f


def audit(rt):
    source,views,c=handles(rt)
    root=Path(rt['run_root'])
    dev='cuda:0'
    rows={}
    parts=[]
    for held in FOLDS:
        f=load_f(root,c,f'rx{held}',dev)
        x,y,rx,ids=role(source,views,c,'V',dev,held=held)
        if len(ids)!=2400 or set(ids)&set(read(root/'F'/f'rx{held}'/'complete.json')['train_ids']):
            raise ValueError('Held source validation rows differ')
        y,rx=tensor(y,dev),tensor(rx,dev)
        cf,p,g,agree=(torch.cat(q) for q in zip(*(view_evidence(f,b) for b in x[0].split(64)),strict=True))
        parts.append((cf,p,g,agree,y,rx))
        rows[SOURCE[held]]=dict(V={v:measure(f,x[i],y) for i,v in enumerate(VIEWS)},
                                confidence=summary(cf,p,g,agree,y))
    cf,p,g,agree,y,rx=(torch.cat(q) for q in zip(*parts,strict=True))
    write(root/'holdout_audit.json',dict(status='SOURCE_HOLDOUT_FROZEN',rows=rows,folds=list(FOLDS),
          confidence=summary(cf,p,g,agree,y),rules=calibrate(p,g,agree,y,rx),source_contract=c,
          U_reads=0,target_reads=0))


def smoke(rt):
    source,views,c=handles(rt)
    with np.load(source/'L_s_metadata.npz',allow_pickle=False) as m:
        y,rx=m['y'],m['rx_i']
        if m['ids'].tolist()!=c['role_ids']['L_s']:
            raise ValueError('True source IDs differ')
    for keep in (np.ones(len(y),dtype=bool),rx!=FOLDS[0]):
        ix=np.concatenate(list(batches(y[keep],rx[keep],np.random.default_rng(721))))
        if not np.array_equal(np.sort(ix),np.arange(int(keep.sum()))):
            raise ValueError('Full source sampler loses or repeats packets')
    f=construct('cpu')
    x=tensor(np.load(views/'L_s.npy',mmap_mode='r',allow_pickle=False)[0,:8])
    logit,_=f(x)
    F.cross_entropy(logit,tensor(y[:8])).backward()
    norm=torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True)
    if logit.shape!=(8,10) or not float(norm)>0:
        raise ValueError('Real aligned source cold start failed')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',L_full=2800,L_fold=2240,target_reads=0))


def wait_source(rt):
    parent=Path(rt['parent_root'])
    deadline=time.monotonic()+4*3600
    while time.monotonic()<deadline:
        try:
            if (parent/'source_views'/'complete.json').is_file():
                handles(rt)
                return
            p=read(parent/'state.json')
        except (FileNotFoundError,json.JSONDecodeError):
            p={}
        if p.get('status')=='FAILED':
            raise RuntimeError(f'Full source view preparation failed: {p}')
        time.sleep(30)
    raise TimeoutError('Full aligned source cache was not committed within four hours')


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state=root/'state.json'
    launch=partial(start,module=__spec__.name)
    try:
        write(state,dict(status='WAITING_FOR_ALIGNED_SOURCE'))
        wait_source(rt)
        write(state,dict(status='SOURCE_SMOKE'),exclusive=False)
        if launch(rt,'smoke','full').wait():
            raise RuntimeError('Full cold supervisor smoke failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(rt,[('train',f'seed{s}') for s in SEEDS]+[('train',f'rx{r}') for r in FOLDS],launch)
        schedule(rt,[('audit','holdout')],launch)
        write(state,dict(status='SOURCE_ONLY_COMPLETE',U_reads=0,target_reads=0),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=('dispatch','smoke','train','audit'))
    p.add_argument('runtime',type=Path)
    p.add_argument('name',nargs='?')
    a=p.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if a.kind=='train':
            train(rt,a.name)
        elif a.kind=='audit':
            audit(rt)
        else:
            smoke(rt)
