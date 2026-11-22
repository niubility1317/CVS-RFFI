"""Matched LoRa PL and receiver-calibrated Tri from one frozen supervisor."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import seed_all
from .data import VIEWS
from .holdout import FOLDS, construct, view_evidence
from .packet_models import SEED
from .packet_runner import schedule
from .refine_f import fused_contrast, labeled_weights, measure
from .refine_publish import HOLDOUT_RUN, SUPERVISED_RUN
from .refine_runner import predict_models, source_handles, start
from .repair_features import complete_labeled_batches
from .runner import activate
from .supervised_extension import metrics, source_arrays

METHODS=('continued_sup','plain_pl','adaptive_tri')
READOUTS={m:('direct',) for m in METHODS}
EPOCHS=48
STEPS=50
BATCH=32


def parent_model(root,c,device):
    parent=root.parent/SUPERVISED_RUN
    dec=read(parent/'source_decision.json')
    folder=parent/'F'/'coherent_practical'
    m=read(folder/'complete.json')
    if not (read(parent/'state.json')['status']=='SOURCE_ONLY_COMPLETE'
            and dec['status']=='SOURCE_SELECTION_FROZEN'
            and dec['selected']=='coherent_practical' and dec['epoch']==120
            and m['method']=='coherent_practical' and m['status']=='SOURCE_FROZEN'
            and m['steps']==6000 and m['source_contract']==c and m['seed']==SEED
            and m['U_reads']==m['target_reads']==0):
        raise ValueError('Exact frozen LoRa supervised E120 family required')
    f=construct(device)
    f.load_state_dict(torch.load(folder/'e120.pt',map_location=device,weights_only=True),strict=True)
    return f


def receiver_audit(root,c):
    h=root.parent/HOLDOUT_RUN
    m=read(h/'holdout_audit.json')
    if (read(h/'state.json')['status']!='SOURCE_ONLY_COMPLETE'
            or m['status']!='SOURCE_HOLDOUT_FROZEN' or m['F_folds']!=list(FOLDS)
            or m['source_contract']!=c or m['seed']!=SEED or m['U_reads'] or m['target_reads']):
        raise ValueError('Expected completed frozen held-receiver audit on this contract')
    return m


def intersect_source(rules, pred, confidence, agree, y, rx):
    accepted=[]
    for row in rules:
        if not row['qualified']:
            continue
        cls,t=row['class_id'],max(.90,row['threshold'])
        take=agree&(pred==cls)&(confidence>=t)
        good=take&(y==cls)
        n,hits=int(take.sum()),int(good.sum())
        n_rx=int((torch.bincount(rx[good],minlength=5)>=2).sum())
        if n>=12 and hits/n>=.90 and n_rx>=3:
            accepted.append(dict(class_id=cls,threshold=t,n=n,hits=hits,receivers=n_rx))
    return accepted if len(accepted)>=2 else []


def policy(rt):
    s,_,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    audit=receiver_audit(root,c)
    with np.load(s/'V_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids']['V']:
            raise ValueError('Teacher source validation rows differ')
        y,rx=tensor(m['y']),tensor(m['rx_i'])
    v=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)[0]
    f=parent_model(root,c,'cpu').eval()
    _,pred,g,agree=(torch.cat(a) for a in zip(*(view_evidence(f,x) for x in tensor(v).split(48)),strict=True))
    rules=intersect_source(audit['rules'],pred,g,agree,y,rx)
    write(root/'source_policy.json',dict(status='SOURCE_POLICY_FROZEN',hard_rules=rules,
          hard_weight=min(.40,.08*len(rules)),heldout_rules=audit['rules'],U_reads=0,target_reads=0))


def load_u(source, cache,c,dev):
    with np.load(source/'U_s_metadata.npz',allow_pickle=False) as m:
        if 'y' in m.files or m['ids'].tolist()!=c['role_ids']['U_s']:
            raise ValueError('The training U source must remain TX-blind')
        rx=tensor(m['rx_i'],dev)
    x=np.load(cache/'U_s.npy',mmap_mode='r',allow_pickle=False)
    if x.shape!=(1,len(rx),6,2,256):
        raise ValueError('Blind U cache shape differs')
    return tensor(x[0],dev),rx


@torch.no_grad()
def cache_teacher(f,u,rules):
    f.eval()
    series=[]
    for b in u.split(48):
        _,p,g,agree=view_evidence(f,b)
        z=f(b)[1]
        kept=torch.zeros_like(agree)
        for r in rules:
            kept|=agree&(p==r['class_id'])&(g>=r['threshold'])
        series.append((p,g,kept,z))
    return tuple(torch.cat(x) for x in zip(*series,strict=True))


def choose(pred,conf,eligible,classes):
    batches=[]
    for cls in classes:
        inds=torch.where(eligible&(pred==cls))[0]
        batches.append(inds[conf[inds].argsort(descending=True)[:3]])
    return torch.cat(batches) if batches else torch.empty(0,dtype=torch.long,device=pred.device)


def rotated(x,generator):
    z=torch.complex(x[:,:,0],x[:,:,1])
    angle=(torch.rand((len(x),1,1),device=x.device,generator=generator)-.5)*.60
    z=z*torch.polar(torch.ones_like(angle),angle)
    return torch.stack((z.real,z.imag),dim=2)


def noise(x,generator):
    return x+.01*torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)


def train(rt,method):
    if method not in METHODS:
        raise ValueError('Unknown LoRa adaptive row')
    source,_,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    folder=root/'F'/method
    dev='cuda:0'
    seed_all(SEED)
    arrays=source_arrays(source,cache,dev)
    f=parent_model(root,c,dev)
    fit(folder,f,method,source,cache,c,arrays,read(root/'source_policy.json'),
        seed=SEED,epochs=EPOCHS,steps=STEPS,sampler=complete_labeled_batches)


def fit(folder,f,method,source,cache,c,arrays,settings,*,seed,epochs,steps,sampler,full=False):
    folder.mkdir(parents=True,exist_ok=False)
    dev='cuda:0'
    (x,y,rx),(v,vy,vr)=arrays
    _,_,pred=measure(f,x[0],y)
    _,weights=labeled_weights(pred,y,True)
    urand=np.random.default_rng(827)
    lrng=np.random.default_rng(721)
    scene_rng=np.random.default_rng(267)
    aug=torch.Generator(device=dev).manual_seed(607)
    y_np,r_np=np.asarray(y.tolist()),np.asarray(rx.tolist())
    if settings['status']!='SOURCE_POLICY_FROZEN':
        raise ValueError('Teacher policy must precede U access')
    classes=[row['class_id'] for row in settings['hard_rules']]
    if method!='continued_sup':
        u,ur=load_u(source,cache,c,dev)
    tri=method in ('adaptive_tri','tri_practical')
    if tri:
        up,uc,uq,uz=cache_teacher(f,u,settings['hard_rules'])
    if method=='tri_practical':
        from .physical import ChirpResponse
        supplied=np.load(cache/'U_s_practical.npy',mmap_mode='r',allow_pickle=False)
        if supplied.shape!=tuple(u.shape) or supplied.dtype!=np.float32:
            raise ValueError('Matched practical U cache differs from registered U')
        practical_u=tensor(supplied,dev)
        ref=ChirpResponse(fft_factor=8).to(dev)
        quality=[]
        with torch.no_grad():
            for batch in practical_u.split(128):
                quality.append(ref.coherence(batch.flatten(0,1)).reshape(len(batch),6).median(1).values)
        usable=torch.cat(quality)>=.40
    opt=torch.optim.AdamW(f.parameters(),lr=1.5e-4,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*steps,eta_min=1.5e-5)
    accepted=[]
    step=0
    fields=('clean_ce','practical_ce','U_ce','U_pair','U_agreement','U_applied')
    with (folder/'epochs.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(1,epochs+1):
            if method!='continued_sup':
                uids=urand.permutation(len(u))[:len(y)]
            total=torch.zeros(len(fields),device=dev)
            used_by_class=torch.zeros(10,dtype=torch.long,device=dev)
            ramp=min(1.,epoch/8.)
            anchors=0
            sat_exposures=0
            for j,ids in enumerate(sampler(y_np,r_np,lrng)):
                idx=tensor(ids,dev); scene=tensor(scene_rng.integers(1,len(VIEWS),size=len(ids)),dev)
                size=len(ids)
                f.train()
                logits,z=f(torch.cat((x[0,idx],x[scene,idx])))
                a,b=logits.chunk(2)
                ce=F.cross_entropy(a,y[idx],weight=weights)
                prac=F.cross_entropy(b,y[idx],weight=weights)
                uce,pair,agree,u_loss=[ce.new_zeros(()) for _ in range(4)]
                if method=='plain_pl':
                    ids_u=tensor(uids[j*BATCH:j*BATCH+size],dev)
                    f.eval()
                    with torch.no_grad():
                        conf,p=f(u[ids_u])[0].softmax(1).max(1)
                    f.train()
                    selected=torch.where(conf>=.95)[0]
                    if len(selected):
                        ul=f(noise(u[ids_u[selected]],aug))[0]
                        uce=F.cross_entropy(ul,p[selected],reduction='sum')/size
                    u_loss=ramp*uce
                elif tri:
                    ids_u=tensor(uids[j*BATCH:j*BATCH+size],dev)
                    p=up[ids_u]
                    selected=choose(p,uc[ids_u],uq[ids_u],classes)
                    uv=u[ids_u]
                    parity=(epoch+j)%2
                    agreement_input=uv
                    if method=='tri_practical':
                        alternating=(torch.arange(size,device=dev)+parity)%2==0
                        use=(usable[ids_u]&alternating)
                        sat_exposures+=int(use.sum())
                        agreement_input=torch.where(use[:,None,None,None],practical_u[ids_u],uv)
                    aligned=agreement_input[:,parity::2].repeat(1,2,1,1)
                    vz=f(noise(rotated(aligned,aug),aug))[1]
                    agree=(1.-F.cosine_similarity(vz,uz[ids_u],dim=1)).mean()
                    if len(selected):
                        ul,feat=f(noise(uv[selected],aug))
                        yy=p[selected]
                        uce=F.cross_entropy(ul,yy,reduction='sum')/size
                        pair,n=fused_contrast(torch.cat((z[:size].detach(),feat)),
                                torch.cat((y[idx],yy)),torch.cat((rx[idx],ur[ids_u[selected]])),u_start=size)
                        anchors+=n
                    u_loss=ramp*(settings['hard_weight']*uce+.025*pair*len(selected)/size+.06*agree)
                if method!='continued_sup' and len(selected):
                    yy=p[selected]
                    accepted.extend(zip([epoch]*len(yy),ids_u[selected].tolist(),yy.tolist()))
                    used_by_class+=torch.bincount(yy,minlength=10)
                opt.zero_grad(set_to_none=True)
                (ce+.35*prac+u_loss).backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True)
                opt.step();sch.step();step+=1
                total+=torch.stack((ce,prac,uce,pair,agree,u_loss)).detach()
            if step!=epoch*steps:
                raise ValueError('Expected exact matched labeled training budget')
            row=dict(epoch=epoch,step=step,loss=dict(zip(fields,(total/steps).tolist(),strict=True)),
                  accepted=used_by_class.tolist(),anchors=anchors,
                  U_exposures=0 if method=='continued_sup' else len(y))
            if full:
                row['U_practical_exposures']=sat_exposures
            if not full or epoch==1 or epoch%4==0 or epoch==epochs:
                row['V']=metrics(f,v[0],vy,vr)
            if epoch in ((1,14,epochs) if full else (1,12,24,36,epochs)):
                row['L']=metrics(f,x[0],y,rx)
                row['V_practical']={name:metrics(f,v[i],vy,vr) for i,name in enumerate(VIEWS) if i}
            record=json.dumps(row,allow_nan=False)
            log.write(record+'\n');log.flush()
            print('LORA_REPAIRED',method,record,flush=True)
    torch.save(f.state_dict(),folder/'model.pt')
    np.save(folder/'accepted_U.npy',np.asarray(accepted,dtype=np.int32).reshape(-1,3))
    write(folder/'complete.json',dict(status='SOURCE_FROZEN',method=method,source_contract=c,seed=seed,
          elapsed_steps=step,epochs=epochs,target_reads=0,U_truth_reads=0))


def load_f(root,c,method,device):
    folder=root/'F'/method
    if read(folder/'complete.json')!=dict(status='SOURCE_FROZEN',method=method,source_contract=c,seed=SEED,
             elapsed_steps=EPOCHS*STEPS,epochs=EPOCHS,target_reads=0,U_truth_reads=0):
        raise ValueError('Adaptive LoRa weights are not source-frozen')
    f=construct(device).eval()
    f.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True),strict=True)
    return f


def source_audit(rt):
    src,_,cache,c=source_handles(rt)
    root=Path(rt['run_root']); dev='cuda:0'
    (x,y,rx),(v,vy,vr)=source_arrays(src,cache,dev)
    records={}
    for method in METHODS:
        f=load_f(root,c,method,dev)
        records[method]=dict(L=metrics(f,x[0],y,rx),V={k:metrics(f,v[i],vy,vr) for i,k in enumerate(VIEWS)})
    write(root/'source_audit.json',dict(status='SOURCE_AUDIT_COMPLETE',records=records,target_reads=0))


def smoke(rt):
    src,_,cache,c=source_handles(rt)
    root=Path(rt['run_root'])
    receiver_audit(root,c)
    model=parent_model(root,c,'cpu').eval()
    a=tensor(np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)[0,:8])
    conf,_,g,_=view_evidence(model,a)
    l,z=model(a)
    with np.load(src/'L_s_metadata.npz',allow_pickle=False) as m:
        if m['ids'].tolist()!=c['role_ids']['L_s']:
            raise ValueError('Source smoke labeled packet order differs')
        y=tensor(m['y'][:8])
    loss=F.cross_entropy(l,y)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    if l.shape!=(8,10) or not float(norm)>0 or not all(torch.isfinite(t).all() for t in (conf,g,l,z)):
        raise ValueError('Frozen supervised parent source-only gradient smoke failed')
    write(root/'smoke.json',dict(status='PASS',gradient=float(norm),U_reads=0,target_reads=0))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch=partial(start,module=__spec__.name)
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        if launch(rt,'smoke','adaptive').wait():
            raise RuntimeError('Adaptive LoRa source smoke failed')
        if launch(rt,'policy','source').wait():
            raise RuntimeError('Adaptive LoRa source policy failed')
        write(state,dict(status='SOURCE_TRAINING'),exclusive=False)
        schedule(rt,[('train',m) for m in METHODS],launch)
        schedule(rt,[('source_audit','all')],launch)
        write(state,dict(status='PREDICTING'),exclusive=False)
        schedule(rt,[('predict',m) for m in METHODS],launch)
        write(state,dict(status='PREDICTIONS_FROZEN',target_truth_read=False),exclusive=False)
    except Exception as e:
        write(state,dict(status='FAILED',error=str(e)),exclusive=False)
        raise


def worker(rt,kind,name):
    activate(rt)
    if kind=='predict':
        _,parent,_,c=source_handles(rt)
        root=Path(rt['run_root'])
        model=load_f(root,c,name,'cuda:0')
        return predict_models(root,parent,Path(rt['target_views']),name,model,{},READOUTS[name])
    if kind=='train':
        return train(rt,name)
    return {'smoke':smoke,'policy':policy,'source_audit':source_audit}[kind](rt)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind')
    parser.add_argument('runtime',type=Path)
    parser.add_argument('name',nargs='?')
    a=parser.parse_args()
    rt=read(a.runtime)
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        worker(rt,a.kind,a.name)
