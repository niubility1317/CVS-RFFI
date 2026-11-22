"""F-free DURA and packet-wise correction dose on matched full-source IQ."""
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .dura_gate import ImpairmentGate, RiskDura, fit_gate
from .dura_repair import complex_iq, no_tone, power, real_iq
from .full_packet_data import COUNTS
from .full_supervised import handles
from .refine_g import aligned_target

GATE_EPOCHS=45
G_STEPS=1600
BATCH=24
DOSE_EPOCHS=50


def iq_arrays(cache,device):
    l=np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)
    u=np.load(cache/'U_s.npy',mmap_mode='r',allow_pickle=False)
    sat=np.load(cache/'U_s_practical.npy',mmap_mode='r',allow_pickle=False)
    n_l,n_u=COUNTS['L_s'],COUNTS['U_s']
    if (l.shape!=(7,n_l,6,2,256) or u.shape!=(1,n_u,6,2,256) or sat.shape!=(n_u,6,2,256)
            or any(q.dtype!=np.float32 for q in (l,u,sat))):
        raise ValueError('Full DURA paired IQ differs')
    return tensor(l,device),tensor(u[0],device),tensor(sat,device)


def gate_record(c):
    return dict(status='SOURCE_FROZEN',source_contract=c,seed=721,epochs=GATE_EPOCHS,
                F_reads=0,TX_reads=0,target_reads=0)


def train_gate(rt,root):
    _,cache,c=handles(rt)
    dev='cuda:0'
    torch.manual_seed(721)
    gate=ImpairmentGate().to(dev)
    l,u,_=iq_arrays(cache,dev)
    def features(x):
        return torch.cat([gate.features(v) for v in x.split(192)])
    p=torch.cat([features(q) for q in l[1:]])
    lc,uc=features(l[0]),features(u)
    h=fit_gate(gate,lc,p,GATE_EPOCHS,721,uc)
    if not np.isfinite(h).all() or any((gate.logits(a)>gate.clean_boundary).any() for a in (lc,uc)):
        raise ValueError('Full source DURA gate training or clean calibration failed')
    root.mkdir(parents=True,exist_ok=False)
    torch.save(gate.state_dict(),root/'model.pt')
    write(root/'training.json',dict(loss=h,clean_L=len(lc),clean_U=len(uc),practical=len(p),
           clean_boundary=float(gate.clean_boundary)))
    write(root/'complete.json',gate_record(c))


def read_gate(root,c,device):
    if read(root/'complete.json')!=gate_record(c):
        raise ValueError('Missing exact source-only DURA gate')
    gate=ImpairmentGate().to(device).eval()
    gate.load_state_dict(torch.load(root/'model.pt',map_location=device,weights_only=True),strict=True)
    return gate


def receiver_loss(error):
    return power(error)+.35*error.mean(1).abs().square().mean(-1)


def protected_shapes(model,x,z):
    """Preserve the observed shared non-carrier residual across the six chirps."""
    w,rot,_,_,_=model.gate.tracker.track(x)
    scale=w.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
    delta=complex_iq(z-x)
    normalized=delta*rot.conj()/scale
    noncarrier=no_tone(normalized)
    shared=noncarrier.mean(1,keepdim=True)
    individual=noncarrier-shared
    limit=power(delta.detach()).sqrt()
    selected=[]
    for component in (normalized-shared,individual):
        correction=component*scale*rot
        norm=power(correction).clamp_min(1e-12).sqrt()
        shrink=(limit/norm).clamp_max(1.)
        selected.append(x+real_iq(correction*shrink[:,None,None]))
    return selected


def dose_oracle(wanted,direction):
    prior=direction.mean(1)
    cov=(wanted.conj()*direction).real.mean((-2,-1)) + .35*(wanted.mean(1).conj()*prior).real.mean(-1)
    norm=receiver_loss(direction)
    return (cov/norm.clamp_min(1e-8)).clamp(0.,1.),norm


def dose_features(model,x,proposal):
    wave=complex_iq(x)
    diff=complex_iq(proposal-x)
    r=power(wave).clamp_min(1e-8).sqrt()
    offset=diff.abs().square().mean(-1).sqrt()/r[:,None]
    obs=model.gate.features(x)
    margin=model.gate.logits(obs)-model.gate.clean_boundary
    q=(obs-model.gate.feature_mean)/model.gate.feature_std
    return torch.cat((q,offset.median(1).values[:,None],offset.max(1).values[:,None],
                      offset.std(1)[:,None],margin[:,None]),dim=1)


class Dose(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(15,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        self.register_buffer('mean',torch.zeros(15))
        self.register_buffer('std',torch.ones(15))

    def forward(self,a):
        return self.body((a-self.mean)/self.std).flatten().sigmoid()


def g_record(c,seed):
    return dict(status='SOURCE_FROZEN',source_contract=c,seed=seed,steps=G_STEPS,
                dose_epochs=DOSE_EPOCHS,cold_start=True,F_reads=0,TX_reads=0,target_reads=0)


def projected_record(c,seed):
    return dict(status='SOURCE_FROZEN',source_contract=c,seed=seed,steps=G_STEPS,
                recipe='protected_shape',cold_start=True,F_reads=0,TX_reads=0,target_reads=0)


def train(rt,root,gate_path,seed,*,projected=False):
    _,cache,c=handles(rt)
    dev='cuda:0'
    torch.manual_seed(seed)
    rng=torch.Generator(device=dev).manual_seed(seed)
    gate=read_gate(gate_path,c,dev)
    model=RiskDura(gate,False).to(dev)
    for p in gate.parameters():
        p.requires_grad_(False)
    l,u,sat=iq_arrays(cache,dev)
    # No physical U row used to fit dosage is used in the G optimizer.
    hold=torch.arange(len(u),device=dev)%5==0
    eligible_U=torch.cat([model.strength(q)>0 for q in sat.split(192)])
    eligible_L=torch.cat([model.strength(q)>0 for q in l[1:].flatten(0,1).split(192)])
    g_l=torch.where(eligible_L)[0]
    g_u=torch.where(eligible_U&~hold)[0]
    d_u=torch.where(eligible_U&hold)[0]
    if min(len(g_l),len(g_u),len(d_u))<64:
        raise ValueError('Insufficient recoverable candidates in G or dose split')
    root.mkdir(parents=True,exist_ok=False)
    opt=torch.optim.AdamW(model.net.parameters(),lr=2e-4,weight_decay=1e-4)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=G_STEPS,eta_min=2e-5)
    record=[]
    with (root/'steps.jsonl').open('x',encoding='utf8') as log:
        for step in range(1,G_STEPS+1):
            # Oversample L views evenly with U (both are IQ-only during G fit).
            j=g_l[torch.randint(len(g_l),(BATCH//2,),device=dev,generator=rng)]
            i=g_u[torch.randint(len(g_u),(BATCH//2,),device=dev,generator=rng)]
            y=torch.cat((l[0,j%len(l[0])],u[i]))
            x=torch.cat((l[j//len(l[0])+1,j%len(l[0])],sat[i]))
            target=complex_iq(aligned_target(y,x,model.reference,'chirp'))
            restored=model(x)
            if projected:
                restored=protected_shapes(model,x,restored)[0]
            error=complex_iq(restored)-target
            loss=receiver_loss(error).mean()+.025*power(complex_iq(restored-x)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            sc.step()
            if step==1 or step%100==0:
                entry=dict(step=step,aligned=float(receiver_loss(error).mean().detach()),
                           raw=float(receiver_loss(complex_iq(x)-target).mean()))
                log.write(json.dumps(entry,allow_nan=False)+'\n')
                log.flush()
                print('FULL_DURA_STEP',seed,entry,flush=True)
    if projected:
        torch.save(model.net.state_dict(),root/'model.pt')
        write(root/'source_support.json',dict(G_L=len(g_l),G_U=len(g_u),reserved_U=len(d_u)))
        write(root/'complete.json',projected_record(c,seed))
        return
    model.eval()
    with torch.no_grad():
        f,a,den=[],[],[]
        for i in d_u.split(32):
            x,y=sat[i],u[i]
            z=model(x)
            clean=complex_iq(aligned_target(y,x,model.reference,'chirp'))
            a_i,v_i=dose_oracle(clean-complex_iq(x),complex_iq(z-x))
            f.append(dose_features(model,x,z))
            a.append(a_i)
            den.append(v_i)
        f,a,weight=[torch.cat(k) for k in (f,a,den)]
    dose=Dose().to(dev)
    dose.mean.copy_(f.mean(0))
    dose.std.copy_(f.std(0).clamp_min(1e-5))
    opt=torch.optim.AdamW(dose.parameters(),lr=2e-3,weight_decay=.01)
    weight=weight/weight.mean().clamp_min(1e-8)
    for _ in range(DOSE_EPOCHS):
        err=f.new_zeros(())
        for j in torch.randperm(len(f),device=dev,generator=rng).split(256):
            loss=(weight[j]*(dose(f[j])-a[j]).square()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            err+=loss.detach()*len(j)
        record.append(float(err/len(f)))
    if not np.isfinite(record).all():
        raise ValueError('DURA dose optimization nonfinite')
    torch.save(dict(G=model.net.state_dict(),dose=dose.state_dict()),root/'model.pt')
    write(root/'dose_training.json',dict(G_L=len(g_l),G_U=len(g_u),dose_U=len(d_u),
          oracle_useful=int((a>0).sum()),oracle_mean=float(a.mean()),weighted_MSE=record))
    write(root/'complete.json',g_record(c,seed))
