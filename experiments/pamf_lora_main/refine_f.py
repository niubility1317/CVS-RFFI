"""Matched LoRa recognizer continuations for spectral input and classwise SSL."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import seed_all
from .data import VIEWS
from .packet_data import ROLE_COUNTS
from .packet_models import SEED, load_source_labels, packet_tri, sym_kl


METHODS = ('raw_pl', 'spectral_pl', 'spectral_focus', 'spectral_tri')
EPOCHS = 18
BATCH = 32
STEPS = 50
PARENT = '20261122-phase1-lora-packet-aligned-pilot-m5-r02'


class SpectralPacket(nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self.register_buffer('window',torch.hann_window(64),persistent=False)
        self.conv = nn.Sequential(nn.Conv2d(6,24,3,padding=1),nn.GroupNorm(4,24),nn.SiLU(),
             nn.Conv2d(24,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
             nn.AdaptiveAvgPool2d((4,3)),nn.Flatten())
        self.head = nn.Linear(384,10)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def spectrum(self,x):
        z = torch.complex(x[:,:,0],x[:,:,1])
        frames = z.unfold(-1,64,16)*self.window
        bins = torch.fft.fftshift(torch.fft.fft(frames,norm='ortho'),dim=-1)[...,15:50]
        # Adjacent-frame log difference cancels a stationary per-frequency gain.
        log_power = bins.abs().square().clamp_min(1e-5).log()
        return (log_power[:,:,1:]-log_power[:,:,:-1]).clamp(-8.,8.).transpose(-1,-2)

    def forward(self,x):
        logits,z = self.raw(x)
        return logits + self.head(self.conv(self.spectrum(x))), z


def source_contract(source):
    c = read(source/'contract.json')
    if (c['protocol']!='lora_packet_pilot_v1' or not c['source_only'] or c['counts']!=ROLE_COUNTS):
        raise ValueError('Same aligned LoRa source contract required')
    return c


def build(parent, contract, method, device, *, load_parent=True):
    if method not in METHODS:
        raise ValueError('Unknown LoRa refinement row')
    f = packet_tri('packet_sup',device)
    if load_parent:
        folder = parent/'F'/'packet_sup'
        c = read(folder/'complete.json')
        if (parent.name!=PARENT or c['status']!='SOURCE_FROZEN' or c['epochs']!=30 or c['seed']!=SEED
                or c['method']!='packet_sup' or c['source_contract']!=contract or c['target_read']):
            raise ValueError('Exact same-contract supervised E30 start required')
        f.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True),strict=True)
    return f if method=='raw_pl' else SpectralPacket(f).to(device)


def class_policy(conf, pred, y, rx):
    result=[]
    for c in range(10):
        chosen=None
        for t in (.60,.65,.70,.75,.80,.85,.90,.95):
            mask=(pred==c)&(conf>=t)
            n=int(mask.sum())
            support=int(rx[mask].unique().numel())
            if n<14 or support<3:
                continue
            hit=int((y[mask]==c).sum())
            p=hit/n
            z=1.96
            low=(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n)
            if low>=.73:
                chosen=dict(class_id=c,threshold=t,n=n,correct=hit,receivers=support,lower=low)
                break
        if chosen:
            result.append(chosen)
    return result


def limit_class(conf,pred,policies):
    selected=[]
    for p in policies:
        ids=torch.where((pred==p['class_id'])&(conf>=p['threshold']))[0]
        if len(ids):
            selected.extend(ids[conf[ids].argsort(descending=True)[:2]].tolist())
    return torch.tensor(selected,dtype=torch.long,device=pred.device)


def fused_contrast(z,y,rx,*,u_start=None):
    same=y[:,None]==y
    cross=rx[:,None]!=rx
    positives=same&cross
    negatives=~same
    if u_start is not None:
        positives[:,u_start:]=False
        negatives[:,u_start:]=False
    valid=positives.any(1)&negatives.any(1)
    if u_start is not None:
        valid[:u_start]=False
    f=F.normalize(z,dim=1)
    logits=(f@f.T/.2).masked_fill(~(positives|negatives),-1e4).log_softmax(1)
    uniform=positives/positives.sum(1,keepdim=True).clamp_min(1)
    weights=uniform
    if u_start is not None:
        donor=positives.float()@(~cross).float()
        balanced=positives/donor.clamp_min(1)
        weights=.5*(uniform+balanced/balanced.sum(1,keepdim=True).clamp_min(1))
    return (-(weights*logits).sum(1)*valid).sum()/valid.sum().clamp_min(1),int(valid.sum())


@torch.no_grad()
def measure(model, inputs, y):
    model.eval()
    score=torch.cat([model(x)[0] for x in inputs.split(32)])
    conf,pred=score.softmax(1).max(1)
    return float((pred==y).float().mean()),conf,pred


def labeled_batches(y, rx, rng):
    cells=np.stack([np.where((y==i)&(rx==j))[0] for i in range(10) for j in range(5)]).reshape(10,5,32)
    for _ in range(STEPS):
        labels=rng.choice(10,8,replace=False)
        receivers=np.stack([rng.choice(5,4,replace=False) for _ in labels])
        yield cells[labels[:,None],receivers,rng.integers(0,32,size=(8,4))].reshape(BATCH)


def labeled_weights(start, y, focus):
    accuracy=torch.stack([(start[y==j]==j).float().mean() for j in range(10)])
    weight=1.+(.5-accuracy).clamp_min(0.) if focus else torch.ones_like(accuracy)
    return accuracy,weight/weight.mean()


def train(source,cache,parent,out,method):
    c=source_contract(source)
    out.mkdir(parents=True,exist_ok=False)
    seed_all(SEED)
    dev='cuda:0'
    x,v,u=(tensor(np.load(cache/f'{r}.npy',mmap_mode='r',allow_pickle=False),dev)
           for r in ('L_s','V','U_s'))
    u=u[0]
    y,rx=load_source_labels(source,'L_s')
    vy,vr=load_source_labels(source,'V')
    with np.load(source/'U_s_metadata.npz',allow_pickle=False) as m:
        if 'y' in m.files or m['ids'].tolist()!=c['role_ids']['U_s']:
            raise ValueError('U source remains TX-blind')
        ur=tensor(m['rx_i'],dev)
    model=build(parent,c,method,dev)
    _,_,start=measure(model,x[0],y)
    by_class,weight=labeled_weights(start,y,method in ('spectral_focus','spectral_tri'))
    write(out/'source_start.json',dict(by_class=by_class.tolist(),weight=weight.tolist(),
                                      checkpoint='packet_sup/E30/own_source'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS*STEPS,eta_min=2e-5)
    lab_rng=np.random.default_rng(721)
    unlab_rng=torch.Generator(device=dev).manual_seed(828)
    label_np=np.asarray(y.tolist())
    rx_np=np.asarray(rx.tolist())
    pseudo_records=[]
    with (out/'epochs.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(1,EPOCHS+1):
            if method=='spectral_tri' and (epoch-1)%3==0:
                _,cv,pv=measure(model,v[0],vy)
                policy=class_policy(cv,pv,vy,vr)
            elif method!='spectral_tri':
                policy=[]
            accepted=torch.zeros(10,dtype=torch.long,device=dev)
            losses=torch.zeros(5,device=dev)
            anchors=0
            gradient=None
            for batch in labeled_batches(label_np,rx_np,lab_rng):
                b=tensor(batch,dev)
                scene=tensor(lab_rng.integers(1,len(VIEWS),size=BATCH),dev)
                model.train()
                logits,z=model(torch.cat((x[0,b],x[scene,b])))
                raw,sat=logits.chunk(2)
                ce=F.cross_entropy(raw,y[b],weight=weight)
                ce_s=F.cross_entropy(sat,y[b],weight=weight)
                xd,_=fused_contrast(z[:BATCH],y[b],rx[b])
                kl=sym_kl(raw,sat).mean()
                u_loss=ce.new_zeros(())
                ui=torch.randint(len(u),(BATCH,),device=dev,generator=unlab_rng)
                model.eval()
                with torch.no_grad():
                    conf,pred=model(u[ui])[0].softmax(1).max(1)
                chosen=limit_class(conf,pred,policy) if method=='spectral_tri' else torch.where(conf>=.95)[0]
                model.train()
                if len(chosen):
                    accepted_ids=ui[chosen]
                    q=u[accepted_ids]
                    yp=pred[chosen]
                    pseudo_records.extend(zip([epoch]*len(chosen),accepted_ids.tolist(),yp.tolist()))
                    noise=torch.randn(q.shape,dtype=q.dtype,device=dev,generator=unlab_rng)
                    ss,uz=model(q+.015*noise)
                    accepted += torch.bincount(yp,minlength=10)
                    if method=='spectral_tri':
                        pair,n=fused_contrast(torch.cat((z[:BATCH].detach(),uz)),torch.cat((y[b],yp)),
                                             torch.cat((rx[b],ur[ui[chosen]])),u_start=BATCH)
                        anchors+=n
                        u_loss=.35*F.cross_entropy(ss,yp,reduction='sum')/max(8,len(chosen))+.025*pair
                    else:
                        u_loss=F.cross_entropy(ss,yp,reduction='sum')/BATCH
                l_loss=ce+.68*ce_s+.10*xd+.10*kl
                if gradient is None and len(chosen):
                    params=tuple(model.parameters())
                    lg,ug=[torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                           for loss in (l_loss,u_loss)]
                    def square_norm(items):
                        return sum(item.square().sum() for item in items if item is not None).sqrt()
                    ln,un=square_norm(lg),square_norm(ug)
                    dot=sum((a*b).sum() for a,b in zip(lg,ug) if a is not None and b is not None)
                    gradient=dict(L_norm=float(ln),U_norm=float(un),cosine=float(dot/(ln*un).clamp_min(1e-12)))
                opt_loss=l_loss+u_loss
                optimizer.zero_grad(set_to_none=True); opt_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();schedule.step()
                losses+=torch.stack([q.detach() for q in (ce,ce_s,xd,kl,u_loss)])
            value,_,pv=measure(model,v[0],vy)
            detail={}
            if epoch in (1,6,12,EPOCHS):
                detail={name:measure(model,v[i],vy)[0] for i,name in enumerate(VIEWS) if i}
            row=dict(epoch=epoch,losses=(losses/STEPS).tolist(),source_clean=value,source_practical=detail,
                     v_class_correct=torch.bincount(vy[pv==vy],minlength=10).tolist(),U_policy=policy,gradient=gradient,
                     U_exposures=BATCH*STEPS,U_accepted_by_class=accepted.tolist(),U_contrast_anchors=anchors)
            print('REFINE_F_EPOCH',method,json.dumps(row,allow_nan=False),flush=True)
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
    torch.save(model.state_dict(),out/'model.pt')
    np.save(out/'accepted_U.npy',np.asarray(pseudo_records,dtype=np.int32).reshape(-1,3))
    write(out/'complete.json',dict(status='SOURCE_FROZEN',method=method,seed=SEED,
                                   epochs=EPOCHS,source_contract=c,target_read=False))
