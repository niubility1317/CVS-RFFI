"""Three independent LoRa seeds with matched supervised, PL and Tri continuations."""
import argparse
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .adaptive_repair import fit, intersect_source
from .comparison import seed_all
from .data import TARGET, VIEWS
from .full_packet_data import LIMIT, PROTOCOL
from .full_restore_draw import RUN as DRAW_RUN
from .full_supervised import FULL_EPOCHS, RUN as SUPERVISED, SEEDS, batches, construct, handles, load_f, role
from .full_tri_repl_lora import (ADAPT_METHODS, ADAPT_RUN, NATIVE_METHODS, REPL_METHODS, REPL_RUN,
                                REPL_SEEDS, WARM_METHODS, WARM_RUN)
from .holdout import FOLDS, summary, view_evidence
from .packet_runner import schedule
from .refine_runner import start
from .runner import activate

RUN='20261122-phase1-lora-full-adaptive-m12-r01'
WARMUP_RUN='20261122-phase1-lora-extended-supervision-m12-r01'
SET_RUN='20261122-phase1-lora-tri-label-set-m3-r01'
SET_METHOD='tri_set'
PRACTICAL_SET_RUN='20261122-phase1-lora-tri-set-practical-draw-m3-r01'
PRACTICAL_SET_METHOD='tri_set_practical'
SET_METHODS=(SET_METHOD,PRACTICAL_SET_METHOD)
METHODS=('continued_sup','plain_pl','adaptive_tri','tri_practical')
WARMUP_METHODS=('continued_sup','plain_pl','adaptive_tri')
EPOCHS=28
STEPS=88
WARMUP_LR=6e-5


def case(seed,method):
    return f's{seed}-{method}'


def cases(methods=METHODS):
    seeds=REPL_SEEDS if tuple(methods) in (REPL_METHODS,WARM_METHODS,ADAPT_METHODS) else SEEDS
    return [(case(s,m),s,m) for s in seeds for m in methods]


def methods_for(root):
    if root.name==ADAPT_RUN:
        return ADAPT_METHODS
    if root.name==WARM_RUN:
        return WARM_METHODS
    if root.name==REPL_RUN:
        return REPL_METHODS
    if root.name==PRACTICAL_SET_RUN:
        return (PRACTICAL_SET_METHOD,)
    if root.name==SET_RUN:
        return (SET_METHOD,)
    return WARMUP_METHODS if root.name==WARMUP_RUN else METHODS


def decode(name):
    value,method=name.split('-',1)
    seed=int(value.removeprefix('s'))
    if seed not in SEEDS or method not in (*METHODS,*SET_METHODS,*NATIVE_METHODS):
        raise ValueError('Unknown full LoRa row')
    return seed,method


def supervisor(root,c):
    path=root.parent/SUPERVISED
    if read(path/'state.json')['status']!='SOURCE_ONLY_COMPLETE':
        raise ValueError('Supervised and held-receiver stages have not finished')
    a=read(path/'holdout_audit.json')
    if (a['status']!='SOURCE_HOLDOUT_FROZEN' or a['folds']!=list(FOLDS)
            or a['source_contract']!=c or a['U_reads'] or a['target_reads']):
        raise ValueError('Full source held-receiver rules differ')
    return path,a


def load_supervised_continuation(root,c,seed,device):
    name=case(seed,'continued_sup')
    path=complete_f(root,c,name)
    rows=[json.loads(line) for line in (path/'epochs.jsonl').read_text().splitlines()]
    if ([r['step'] for r in rows]!=list(range(STEPS,EPOCHS*STEPS+1,STEPS))
            or any(r['U_exposures'] or r['anchors'] or any(r['accepted']) for r in rows)):
        raise ValueError('The teacher ancestor must be purely supervised')
    with (path/'accepted_U.npy').open('rb') as fp:
        if np.load(fp,allow_pickle=False).shape!=(0,3):
            raise ValueError('The teacher ancestor has saved U decisions')
    model=construct(device).eval()
    model.load_state_dict(torch.load(path/'model.pt',map_location=device,weights_only=True),strict=True)
    return model


def teacher(root,c,seed,device):
    parent,_=supervisor(root,c)
    if root.name in (SET_RUN,PRACTICAL_SET_RUN):
        return load_supervised_continuation(root.parent/WARMUP_RUN/'warmup',c,seed,device)
    if root.name==WARMUP_RUN:
        return load_supervised_continuation(root/'warmup',c,seed,device)
    return load_f(parent,c,f'seed{seed}',device)


def source_arrays(src,views,c,dev):
    data=[]
    for key in ('L_s','V'):
        x,y,rx,_=role(src,views,c,key,dev)
        data.append((x,tensor(y,dev),tensor(rx,dev)))
    return data


def warmup(rt,seed):
    src,views,c=handles(rt)
    root=Path(rt['run_root'])
    supervisor(root,c)
    previous=root.parent/RUN
    if read(previous/'source_audit.json')['status']!='SOURCE_AUDIT_COMPLETE':
        raise ValueError('The matched supervised ancestor has no completed source audit')
    seed_all(seed)
    dev='cuda:0'
    f=load_supervised_continuation(previous,c,seed,dev)
    settings=dict(status='SOURCE_POLICY_FROZEN',hard_rules=[],hard_weight=0.)
    fit(root/'warmup'/'F'/case(seed,'continued_sup'),f,'continued_sup',src,views,c,
        source_arrays(src,views,c,dev),settings,seed=seed,epochs=EPOCHS,steps=STEPS,
        sampler=batches,full=True,lr=WARMUP_LR)


def policy(rt,seed):
    src,views,c=handles(rt)
    root=Path(rt['run_root'])
    if root.name in (SET_RUN,PRACTICAL_SET_RUN):
        frozen=read(root.parent/WARMUP_RUN/'policy'/f'{seed}.json')
        if (frozen['status']!='SOURCE_POLICY_FROZEN' or frozen['seed']!=seed
                or frozen['supervised_epochs']!=FULL_EPOCHS+2*EPOCHS
                or frozen['U_reads'] or frozen['target_reads']):
            raise ValueError('Set comparison requires the same frozen E140 hard policy')
        write(root/'policy'/f'{seed}.json',frozen)
        return
    _,a=supervisor(root,c)
    dev='cuda:0'
    f=teacher(root,c,seed,dev)
    x,y,rx,_=role(src,views,c,'V',dev)
    y,rx=tensor(y,dev),tensor(rx,dev)
    conf,p,g,agree=(torch.cat(q) for q in zip(*(view_evidence(f,b) for b in x[0].split(64)),strict=True))
    rules=intersect_source(a['rules'],p,g,agree,y,rx)
    record=dict(status='SOURCE_POLICY_FROZEN',hard_rules=rules,hard_weight=min(.40,.08*len(rules)),
                heldout_rules=a['rules'],seed=seed,U_reads=0,target_reads=0)
    if root.name==WARMUP_RUN:
        supervised_epochs=FULL_EPOCHS+2*EPOCHS
        record.update(supervised_epochs=supervised_epochs,supervised_steps=supervised_epochs*STEPS,
                      teacher_clean_confidence=summary(conf,p,g,agree,y))
    write(root/'policy'/f'{seed}.json',record)


def train(rt,name):
    seed,method=decode(name)
    if method in NATIVE_METHODS:
        from .full_tri_repl_lora import train as train_repl
        return train_repl(rt,seed,method)
    src,views,c=handles(rt)
    root=Path(rt['run_root'])
    dev='cuda:0'
    seed_all(seed)
    f=teacher(root,c,seed,dev)
    data=source_arrays(src,views,c,dev)
    settings=read(root/'policy'/f'{seed}.json')
    if settings['seed']!=seed or settings['U_reads'] or settings['target_reads']:
        raise ValueError('Full LoRa qualification policy is not source-only')
    alternate=root.parent/DRAW_RUN if method==PRACTICAL_SET_METHOD else None
    fit(root/'F'/name,f,method,src,views,c,data,settings,
        seed=seed,epochs=EPOCHS,steps=STEPS,sampler=batches,full=True,
        lr=WARMUP_LR if root.name in (WARMUP_RUN,SET_RUN,PRACTICAL_SET_RUN) else 1.5e-4,
        alternate_path=alternate)


def complete_f(root,c,name):
    seed,method=decode(name)
    path=root/'F'/name
    if method in NATIVE_METHODS:
        from .full_tri_repl_lora import record
        if seed not in REPL_SEEDS or read(path/'complete.json')!=record(c,seed,method):
            raise ValueError('Original LoRa Tri source freeze differs')
        return path
    if read(path/'complete.json')!=dict(status='SOURCE_FROZEN',method=method,source_contract=c,seed=seed,
             elapsed_steps=EPOCHS*STEPS,epochs=EPOCHS,target_reads=0,U_truth_reads=0):
        raise ValueError('The complete registered weights did not freeze')
    return path


def audit(rt):
    _,_,c=handles(rt)
    root=Path(rt['run_root'])
    results={}
    for name,seed,m in cases(methods_for(root)):
        path=complete_f(root,c,name)
        a=[json.loads(l) for l in (path/'epochs.jsonl').read_text().splitlines()]
        steps=read(path/'complete.json')['elapsed_steps'] if m in NATIVE_METHODS else STEPS*EPOCHS
        if [r['step'] for r in a]!=list(range(STEPS,steps+1,STEPS)):
            raise ValueError('Incomplete full-source record')
        row=a[-1]
        if m in NATIVE_METHODS:
            warm=read(path/'complete.json').get('additional_supervised_epochs',0)
            if any(t['U_seen'] or t['U_accepted'] for t in a[:warm]):
                raise ValueError('Original Tri warmup accessed U before enabling REPL')
            results[name]=dict(seed=seed,method=m,L=row['L'],V=row['V'],
                hard_U_count=sum(t['U_accepted'] for t in a),U_total=sum(t['U_seen'] for t in a),
                style_U_paired=sum(t['U_partners'] for t in a),forwards=sum(t['U_forwards'] for t in a),
                practical_U_seen=sum(t.get('U_practical',0) for t in a))
            continue
        results[name]=dict(seed=seed,method=m,L=row['L'],V={'clean':row['V'],**row['V_practical']},
              hard_U_count=sum(sum(t['accepted']) for t in a),
              cross_rx_anchors=sum(t['anchors'] for t in a),
              practical_U_uses=sum(t['U_practical_exposures'] for t in a))
        if m in SET_METHODS:
            results[name]['set_U_count']=sum(t['U_set_exposures'] for t in a)
        if m==PRACTICAL_SET_METHOD:
            results[name]['alternate_U_count']=sum(t['U_alternate_exposures'] for t in a)
    write(root/'source_audit.json',dict(status='SOURCE_AUDIT_COMPLETE',target_reads=0,records=results))


def predict(rt,name):
    _,_,c=handles(rt)
    root=Path(rt['run_root'])
    data=Path(rt['parent_root'])
    t=Path(rt['target_views'])
    if (read(data/'state.json')['status']!='FULL_VIEWS_READY' or t!=data/'target_views'
            or read(t/'complete.json')!=dict(status='TARGET_READY',protocol=PROTOCOL,target_truth_read=False,
                              receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS))):
        raise ValueError('Only complete registered blind target IQ can be predicted')
    seed,method=decode(name)
    path=complete_f(root,c,name)
    dev='cuda:0'
    if method in NATIVE_METHODS:
        from .full_tri_repl_lora import backbone
        model=backbone(method,dev).eval()
    else:
        model=construct(dev).eval()
    model.load_state_dict(torch.load(path/'model.pt',map_location=dev,weights_only=True),strict=True)
    out=root/'predictions'/name
    out.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():
        for rx in TARGET:
            a=np.load(t/f'{rx}.npy',allow_pickle=False,mmap_mode='r')
            if a.shape!=(len(VIEWS),LIMIT,6,2,256) or a.dtype!=np.float32:
                raise ValueError('Target physical LoRa view matrix differs')
            for j,v in enumerate(VIEWS):
                pp=[]
                for b in tensor(a[j],dev).split(96):
                    score=model(b)[0]
                    if not torch.isfinite(score).all():
                        raise ValueError('Nonfinite target LoRa logits')
                    pp.extend(score.argmax(1).tolist())
                np.save(out/f'{rx}__{v}__direct.npy',np.asarray(pp,dtype=np.int16))
            print('FULL_LORA_BLIND_PREDICTION',name,rx,flush=True)
    write(out/'complete.json',dict(status='PREDICTIONS_FROZEN',seed=seed,method=method,
          target_truth_read=False,receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS),modes=['direct']))


def smoke(rt):
    if Path(rt['run_root']).name in (REPL_RUN,WARM_RUN,ADAPT_RUN):
        from .full_tri_repl_lora import smoke as smoke_repl
        return smoke_repl(rt)
    src,views,c=handles(rt)
    root=Path(rt['run_root'])
    supervisor(root,c)
    a=tensor(np.load(views/'L_s.npy',mmap_mode='r',allow_pickle=False)[0,:8])
    with np.load(src/'L_s_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids']['L_s']:
            raise ValueError('Unbound full source smoke IQ')
        labels=tensor(m['y'][:8])
    receipt={}
    for seed in SEEDS:
        model=(load_supervised_continuation(root.parent/RUN,c,seed,'cpu') if root.name==WARMUP_RUN
               else teacher(root,c,seed,'cpu'))
        result,features=model(a)
        torch.nn.functional.cross_entropy(result,labels).backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
        if result.shape!=(8,10) or not grad>0 or not torch.isfinite(features).all():
            raise ValueError('Real full LoRa supervisor source smoke failed')
        receipt[str(seed)]=grad
    write(root/'smoke.json',dict(status='PASS',grad=receipt,U_reads=0,target_reads=0))


def wait_for(path,want):
    until=time.monotonic()+8*3600
    while time.monotonic()<until:
        try:
            x=read(path/'state.json')
        except (FileNotFoundError,json.JSONDecodeError):
            x={}
        if x.get('status')==want:
            return
        if x.get('status')=='FAILED':
            raise RuntimeError(f'Required LoRa predecessor failed: {path.name} {x}')
        time.sleep(30)
    raise TimeoutError(f'No complete predecessor {path.name} after eight hours')


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch=partial(start,module=__spec__.name)
    state=root/'state.json'
    try:
        write(state,dict(status='WAITING_FOR_SOURCE_TEACHERS'))
        wait_for(root.parent/SUPERVISED,'SOURCE_ONLY_COMPLETE')
        if root.name in (SET_RUN,PRACTICAL_SET_RUN,REPL_RUN,ADAPT_RUN):
            wait_for(root.parent/WARMUP_RUN,'PREDICTIONS_FROZEN')
        if root.name==PRACTICAL_SET_RUN:
            wait_for(root.parent/DRAW_RUN,'SOURCE_DRAW_READY')
            wait_for(root.parent/SET_RUN,'PREDICTIONS_FROZEN')
        write(state,dict(status='SOURCE_SMOKE'),exclusive=False)
        if launch(rt,'smoke','full').wait():
            raise RuntimeError('Actual full-parent source smoke failed')
        if root.name==WARMUP_RUN:
            write(state,dict(status='SUPERVISION_EXTENSION'),exclusive=False)
            schedule(rt,[('warmup',str(s)) for s in SEEDS],launch)
        if root.name not in (REPL_RUN,WARM_RUN,ADAPT_RUN):
            schedule(rt,[('policy',str(s)) for s in SEEDS],launch)
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        names=[n for n,_,_ in cases(methods_for(root))]
        schedule(rt,[('train',name) for name in names],launch)
        if launch(rt,'audit','full').wait():
            raise RuntimeError('Full source log audit failed')
        write(state,dict(status='WAITING_FOR_BLIND_VIEWS'),exclusive=False)
        wait_for(Path(rt['parent_root']),'FULL_VIEWS_READY')
        write(state,dict(status='PREDICTING'),exclusive=False)
        schedule(rt,[('predict',name) for name in names],launch)
        write(state,dict(status='PREDICTIONS_FROZEN',target_truth_read=False),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=('dispatch','smoke','warmup','policy','train','audit','predict'))
    p.add_argument('runtime',type=Path)
    p.add_argument('name',nargs='?')
    a=p.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if a.kind=='policy':
            policy(rt,int(a.name))
        elif a.kind=='warmup':
            warmup(rt,int(a.name))
        elif a.kind in ('train','predict'):
            {'train':train,'predict':predict}[a.kind](rt,a.name)
        else:
            {'smoke':smoke,'audit':audit}[a.kind](rt)
