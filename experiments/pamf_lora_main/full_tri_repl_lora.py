"""LoRa packet adaptation of the original six-pass Tri-SIG-TIME REPL objective."""
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.f_model import contrast
from experiments.cric_dura_comparison.io import tensor, write
from experiments.pamf_ssl_research.hard_pl_mixstyle_objective import style_hooks
from experiments.pamf_ssl_research.hard_pl_objective import mechanism_loss
from experiments.pamf_ssl_research.source_receiver_extrapolation import extrapolate
from experiments.pamf_ssl_research.u_gradprotect_objective import protected_loss
from experiments.pamf_ssl_research.u_mixstyle_objective import protected_partners
from .adaptive_repair import load_u
from .comparison import seed_all
from .data import SOURCE, VIEWS
from .full_supervised import SEEDS, batches, handles


REPL_RUN='20261122-phase1-lora-native-tri-repl-m2-r02'
REPL_SEEDS=(SEEDS[1],)
REPL_METHODS=('native_repl6','native_repl3')
U_WEIGHTS=dict(zip(REPL_METHODS,(6.,3.),strict=True))
EPOCHS=28
STEPS=88
U_BATCH=64
STYLE=dict(levels=('time_down','t2.pool'),alpha=.1,probability_per_level=.5,eps=1e-6)
PREFIX='encoder.core.id_backbone.'


def ordinary_iq(raw):
    return raw+.015*torch.randn_like(raw)


def packet_outputs(model,x,style=None,*,capture=False):
    maps={}
    handles=[]

    def save(name):
        def hook(_,args,out):
            maps[name]=out.reshape(len(x),x.shape[1],*out.shape[1:]).mean(1)
        return hook

    try:
        if capture:
            for name in STYLE['levels']:
                handles.append(model.get_submodule(PREFIX+name).register_forward_hook(save(name)))
        context=nullcontext() if style is None else style_hooks(
            model,PREFIX,STYLE,*style[:3],enabled_by_level=style[3],prepend=True)
        with context:
            logits,h=model(x)
    finally:
        for handle in handles:
            handle.remove()
    if capture and set(maps)!=set(STYLE['levels']):
        raise ValueError('Original shallow and middle capture did not observe a packet')
    return logits,h,maps


def six_views(model,raw,mixed,ordinary,style):
    return [packet_outputs(model,x,style,capture=i<2)
            for i,x in enumerate((ordinary,ordinary,mixed,mixed,raw,raw))]


def mean_contrast(outputs,y,rx,keep,*,joint=False):
    losses=[]
    for _,h,m in outputs[:2]:
        maps={'joint160':h} if joint else m
        subset={k:x[keep] for k,x in maps.items()}
        if keep.any():
            losses.append(contrast(subset,y[keep],rx[keep])[0])
        else:
            losses.append(h.sum()*0.)
    return torch.stack(losses).mean()


def labeled_loss(model,x,practical,y,rx,step):
    ordinary=ordinary_iq(x)
    use=(torch.arange(len(x),device=x.device)+step)%2==0
    mixed=torch.where(use[:,None,None,None],practical,ordinary)
    outputs=six_views(model,x,mixed,ordinary,None)
    accepted=torch.ones_like(use)
    aux=mean_contrast(outputs,y,rx,accepted)
    loss,detail=mechanism_loss(*(p[0] for p in outputs),y,accepted,use,aux)
    return loss,detail


def unlabeled_loss(model,x,rx,step,weight,*,forced_pseudo=None):
    model.eval()
    with torch.no_grad():
        probabilities=model(x)[0].softmax(1)
        confidence,y=probabilities.max(1)
        accepted=confidence>=.95
        if forced_pseudo is not None:
            y=forced_pseudo
            accepted=torch.ones_like(accepted)
            probabilities=F.one_hot(y,10).to(x.dtype)
        partner,valid=protected_partners(probabilities,accepted.float(),rx,range(len(SOURCE)),1.)
        n_chirp=x.shape[1]
        rows=partner[:,None]*n_chirp+torch.arange(n_chirp,device=x.device)
        alpha=torch.tensor(STYLE['alpha'],device=x.device)
        mix=torch.distributions.Beta(alpha,alpha).sample((len(x),1,1)).repeat_interleave(n_chirp,0)
        enabled={name:bool(torch.rand((),device=x.device)<.5) for name in STYLE['levels']}
        if forced_pseudo is not None:
            enabled=dict.fromkeys(STYLE['levels'],True)
        style=(rows.flatten(),valid.repeat_interleave(n_chirp),mix,enabled)
        ordinary=ordinary_iq(x)
        use=(torch.arange(len(x),device=x.device)+step)%2==0
        mixed=torch.where(use[:,None,None,None],x,ordinary)
    model.train()
    outputs=six_views(model,x,mixed,ordinary,style)
    shallow=mean_contrast(outputs,y,rx,accepted)
    loss,detail=mechanism_loss(*(p[0] for p in outputs),y,accepted,use,shallow)
    joint=mean_contrast(outputs,y,rx,accepted,joint=True)
    replacing=[]
    for logits,h,_ in outputs[:2]:
        h_ext=extrapolate(h,y,accepted,rx,partner,valid,.25)
        logits_ext=model.encoder.classify_features(h_ext)
        d=F.cross_entropy(logits_ext,y,reduction='none',label_smoothing=.01)
        e=F.cross_entropy(logits,y,reduction='none',label_smoothing=.01)
        replacing.append(((d-e)*valid).mean())
    replace=torch.stack(replacing).mean()
    loss=weight*(loss+(.05*joint+.25*replace)/1.25)
    return loss,y,accepted,dict(detail,joint=joint.detach(),extrapolation=replace.detach(),
                               partners=int(valid.sum()),forwards=len(outputs))


def initialized(root,c,seed,device):
    from .full_adaptive import WARMUP_RUN, load_supervised_continuation
    warm=load_supervised_continuation(root.parent/WARMUP_RUN/'warmup',c,seed,device)
    return warm.raw


def record(c,seed,method):
    return dict(status='SOURCE_FROZEN',source_contract=c,seed=seed,method=method,epochs=EPOCHS,
                elapsed_steps=EPOCHS*STEPS,source_parent='E140_supervised_original_Tri_branch',
                F_head='default_Tri',U_weight=U_WEIGHTS[method],U_truth_reads=0,target_reads=0)


@torch.no_grad()
def accuracy(model,x,y):
    model.eval()
    p=torch.cat([model(v)[0] for v in x.split(64)])
    return dict(n=len(y),correct=int((p.argmax(1)==y).sum()),ce=float(F.cross_entropy(p,y)))


def train(rt,seed,method):
    from .full_adaptive import source_arrays
    if seed not in REPL_SEEDS or method not in REPL_METHODS:
        raise ValueError('Unregistered original LoRa Tri run row')
    root=Path(rt['run_root'])
    src,views,c=handles(rt)
    dev='cuda:0'
    seed_all(seed)
    model=initialized(root,c,seed,dev)
    (x,y,rx),(v,vy,_)=source_arrays(src,views,c,dev)
    u,ur=load_u(src,views,c,dev)
    yn,rn=np.asarray(y.tolist()),np.asarray(rx.tolist())
    out=root/'F'/f's{seed}-{method}'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'initial_source.json',dict(L=accuracy(model,x[0],y),
          V={name:accuracy(model,v[i],vy) for i,name in enumerate(VIEWS)},target_reads=0))
    opt=torch.optim.Adam(model.parameters(),lr=6e-5,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS*STEPS,eta_min=1e-5)
    named=[(k,p) for k,p in model.named_parameters() if p.requires_grad]
    rng=np.random.default_rng(721)
    accepted_rows=[]
    total=0
    with (out/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,EPOCHS+1):
            loss_l=loss_u=0.
            seen=accepted=paired=forward=0
            gradients=[]
            u_order=torch.randperm(len(u),device=dev)[:STEPS*U_BATCH].reshape(STEPS,U_BATCH)
            for ids,pos in zip(batches(yn,rn,rng),u_order,strict=True):
                i=tensor(ids,dev)
                total+=1
                scene=(i+total+epoch)%6+1
                model.train()
                opt.zero_grad(set_to_none=True)
                lab,_=labeled_loss(model,x[0,i],x[scene,i],y[i],rx[i],total)
                unlab,py,keep,extra=unlabeled_loss(model,u[pos],ur[pos],total,U_WEIGHTS[method])
                loss,docket,_=protected_loss(lab,unlab,named)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step()
                sched.step()
                loss_l+=float(lab.detach())
                loss_u+=float(unlab.detach())
                seen+=len(pos)
                accepted+=int(keep.sum())
                paired+=extra['partners']
                forward+=extra['forwards']
                values=torch.stack((torch.full_like(pos[keep],epoch),pos[keep],py[keep]),dim=1)
                accepted_rows.append(np.asarray(values.tolist(),dtype=np.int64).reshape(-1,3))
                gradients.append(docket)
            validation=None
            if epoch==1 or epoch%4==0:
                validation={name:accuracy(model,v[j],vy) for j,name in enumerate(VIEWS)}
            row=dict(epoch=epoch,step=total,L_loss=loss_l/STEPS,U_loss=loss_u/STEPS,
                     U_seen=seen,U_accepted=accepted,U_partners=paired,U_forwards=forward,
                     L=accuracy(model,x[0],y),V=validation,
                     gradients={group:{k:sum(item[group][k] for item in gradients)/len(gradients)
                                       for k in values} for group,values in gradients[0].items()})
            if total!=epoch*STEPS:
                raise ValueError('LoRa native Tri source batch budget differs')
            log.write(json.dumps(row,allow_nan=False)+'\n')
            log.flush()
            print('LORA_NATIVE_REPL',seed,method,epoch,total,row['L'],
                  None if validation is None else validation['clean'],accepted,flush=True)
    torch.save(model.state_dict(),out/'model.pt')
    np.save(out/'accepted_U.npy',np.concatenate(accepted_rows))
    write(out/'complete.json',record(c,seed,method))


def smoke(rt):
    from .full_supervised import role
    root=Path(rt['run_root'])
    source,views,c=handles(rt)
    seed,=REPL_SEEDS
    seed_all(seed)
    model=initialized(root,c,seed,'cpu').train()
    x,y,rx,_=role(source,views,c,'L_s','cpu')
    idx=next(batches(y,rx,np.random.default_rng(721)))[:8]
    row=tensor(idx)
    labels,receivers=tensor(y[idx]),tensor(rx[idx])
    lab,_=labeled_loss(model,x[0,row],x[1,row],labels,receivers,1)
    unlab,_,accepted,terms=unlabeled_loss(model,x[0,row],receivers,1,6.,forced_pseudo=labels)
    optimizer=torch.optim.Adam(model.parameters(),lr=6e-5)
    named=[(k,p) for k,p in model.named_parameters() if p.requires_grad]
    guarded,groups,_=protected_loss(lab,unlab,named)
    guarded.backward()
    g=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
    if not (g>0 and terms['forwards']==6 and terms['partners']>0 and accepted.all()):
        raise ValueError('Original native Tri six-pass source smoke failed')
    optimizer.step()
    write(root/'smoke.json',dict(status='PASS',grad=g,actual_E140=True,style_pairs=terms['partners'],
                                grad_groups=groups,U_reads=0,U_truth_reads=0,target_reads=0))
