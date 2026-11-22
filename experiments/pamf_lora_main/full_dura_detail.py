"""Learn LoRa noncarrier detail on repeated chirps from source IQ pairs."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import TARGET, VIEWS
from .dura_repair import complex_iq, no_tone, power, real_iq
from .full_dura_core import iq_arrays, protected_shapes, read_gate
from .full_dura_stage import alternate_arrays
from .full_packet_data import LIMIT
from .full_restore_draw import RUN as DRAW_RUN, load_draw
from .full_supervised import SEEDS, handles


DETAIL_RUN='20261122-phase1-lora-dura-coherent-detail-m1-r01'
DETAIL_DOSE_RUN='20261122-phase1-lora-dura-coherent-dose-m2-r01'
DETAIL_CONFIRM_RUN='20261122-phase1-lora-dura-coherent-dose-confirm-m4-r01'
DETAIL_SEEDS=(SEEDS[1],)
CONFIRM_NEW_SEEDS=(SEEDS[0],SEEDS[2])
DETAIL_MODES=('direct','fixed','protect_shape','coherent_detail')
DETAIL_DOSES={'coherent_quarter':.25,'coherent_half':.50}
DETAIL_DOSE_MODES=(*DETAIL_MODES,*DETAIL_DOSES)
STEPS=1800
CAP=.15
QUALITY=.94


def conv_head(channels):
    return nn.Sequential(nn.Conv1d(channels,32,9,padding=4,padding_mode='reflect'),nn.SiLU(),
        nn.Conv1d(32,32,9,padding=8,dilation=2,padding_mode='reflect'),nn.SiLU(),
        nn.Conv1d(32,2,9,padding=4,padding_mode='reflect'))


def pair_channels(z):
    return torch.stack((z.real,z.imag),dim=-2)


def strength_ratio(delta,x):
    return power(delta).clamp_min(1e-12).sqrt()/power(x).clamp_min(1e-12).sqrt()


class PacketDetail(nn.Module):
    """Predict separate common and individual shape corrections in carrier coordinates."""

    def __init__(self,gate):
        super().__init__()
        self.gate=gate.requires_grad_(False)
        self.common=conv_head(3)
        self.individual=conv_head(5)
        for head in (self.common,self.individual):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.utility=nn.Sequential(nn.Linear(13,24),nn.SiLU(),nn.Linear(24,1))
        self.register_buffer('center',torch.zeros(13))
        self.register_buffer('scale',torch.ones(13))
        self.register_buffer('threshold',torch.tensor(float('inf')))

    @torch.no_grad()
    def eligible(self,x):
        tracker=self.gate.tracker
        return tracker.reference.coherence(x.flatten(0,1)).reshape(len(x),6).median(1).values>=QUALITY

    @torch.no_grad()
    def coordinates(self,x):
        w,rot,_,_,_=self.gate.tracker.track(x)
        rms=w.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
        return no_tone(w/rms),rot,rms

    def correction(self,shape,rot,rms):
        common=shape.mean(1)
        own=shape-common[:,None]
        spread=own.abs().square().mean(1).clamp_min(1e-8).sqrt()[:,None]
        shared=torch.cat((pair_channels(common),spread),dim=1)
        individual=torch.cat((pair_channels(own),shared[:,None].expand(-1,6,-1,-1)),dim=2)
        s=self.common(shared)
        i=self.individual(individual.flatten(0,1)).reshape(len(shape),6,2,256)
        a=no_tone(torch.complex(s[:,0],s[:,1]))
        b=no_tone(torch.complex(i[:,:,0],i[:,:,1]))
        b=b-b.mean(1,keepdim=True)
        delta=(a[:,None]+b)*rms
        ratio=strength_ratio(delta,rms.expand_as(delta))
        shrink=(CAP/ratio).clamp_max(1.)[:,None,None]
        return (a[:,None]+b)*shrink,real_iq(delta*shrink*rot)

    def proposal(self,x):
        return self.correction(*self.coordinates(x))[1]

    def features(self,x,delta):
        d=complex_iq(delta)
        r=strength_ratio(d,complex_iq(x))
        offset=d.abs().square().mean(-1).sqrt()
        variation=offset.std(1)/offset.mean(1).clamp_min(1e-6)
        return torch.cat((self.gate.features(x),r[:,None],variation[:,None]),dim=1)

    def score(self,features):
        return self.utility((features-self.center)/self.scale).flatten()

    def accepted(self,x,delta):
        return self.score(self.features(x,delta))>self.threshold


def shape_error(error):
    return power(error)+.5*error.mean(1).abs().square().mean(-1)


def training_terms(model,x,clean):
    shape,rot,rms=model.coordinates(x)
    target,_,_=model.coordinates(clean)
    step,delta=model.correction(shape,rot,rms)
    raw=shape_error(shape-target)
    error=shape_error(shape+step-target)
    return raw,error,delta


def record(c,seed):
    return dict(status='SOURCE_FROZEN',source_contract=c,seed=seed,steps=STEPS,
                draw_run=DRAW_RUN,quality=QUALITY,cap=CAP,F_reads=0,TX_reads=0,target_reads=0)


def source_inputs(rt,device):
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    return c,iq_arrays(cache,device),alternate_arrays(root,device)


def tracker_gate(root,c,device):
    from .full_dura import RUN
    return read_gate(root.parent/RUN/'G/gate',c,device)


@torch.no_grad()
def pairs_for_gate(model,clean,practical):
    features=[]
    improvement=[]
    for a,x in zip(clean.split(128),practical.split(128),strict=True):
        keep=model.eligible(x)
        if keep.any():
            x=x[keep]
            raw,error,d=training_terms(model,x,a[keep])
            features.append(model.features(x,d))
            improvement.append((raw-error)/raw.clamp_min(.0001))
    return torch.cat(features),torch.cat(improvement)


@torch.no_grad()
def clean_features(model,clean):
    features=[]
    for x in clean.split(128):
        keep=model.eligible(x)
        if keep.any():
            x=x[keep]
            features.append(model.features(x,model.proposal(x)))
    return torch.cat(features)


def fit_utility(model,l,u,sat,alt,held):
    fit=held[::2]
    check=held[1::2]
    observations=[pairs_for_gate(model,u[fit],a[fit]) for a in (sat,alt)]
    clean_fit=clean_features(model,u[fit])
    x=torch.cat((clean_fit,*(f for f,_ in observations)))
    label=torch.cat((x.new_zeros(len(clean_fit)),*((e>.05).float() for _,e in observations)))
    model.center.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp_min(1e-5))
    opt=torch.optim.AdamW(model.utility.parameters(),lr=.004,weight_decay=.01)
    history=[]
    for _ in range(30):
        losses=[]
        for i in torch.randperm(len(x),device=u.device).split(512):
            z=label[i]
            loss=F.binary_cross_entropy_with_logits(model.score(x[i]),z,weight=torch.where(z==0,2.,1.))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    with torch.no_grad():
        top=max(model.score(clean_features(model,a)).max() for a in (l,u))
        model.threshold.copy_(top+1e-4)
        audits=[]
        for a in (sat,alt):
            f,e=pairs_for_gate(model,u[check],a[check])
            picked=model.score(f)>model.threshold
            audits.append(dict(high_quality=len(f),selected=int(picked.sum()),beneficial=int((e[picked]>0).sum()),
                                relative_change_sum=float(e[picked].sum())))
    return dict(loss=history,fit_n=len(x),beneficial_fit_n=int(label.sum()),threshold=float(model.threshold),
                withheld_U_pairs=audits)


def train_detail(rt,seed):
    root=Path(rt['run_root'])
    seeds=CONFIRM_NEW_SEEDS if root.name==DETAIL_CONFIRM_RUN else DETAIL_SEEDS
    if seed not in seeds:
        raise ValueError('Unregistered high-coherence DURA seed')
    dev='cuda:0'
    torch.manual_seed(seed)
    c,(l,u,sat),(alt_l,alt_u)=source_inputs(rt,dev)
    model=PacketDetail(tracker_gate(root,c,dev)).to(dev)
    held=torch.arange(0,len(u),5,device=dev)
    signals=(l[1:].flatten(0,1),sat,alt_l,alt_u)
    sources=(l[0],u,l[0],u)
    masks=[torch.cat([model.eligible(x) for x in data.split(256)]) for data in signals]
    for i in (1,3):
        masks[i][held]=False
    pools=[torch.where(mask)[0] for mask in masks]
    if min(map(len,pools))<24:
        raise ValueError('Insufficient high-coherence source support')
    out=root/'G'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    parameters=[*model.common.parameters(),*model.individual.parameters()]
    opt=torch.optim.AdamW(parameters,lr=.0006,weight_decay=.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00006)
    counts=(6,8,4,6)
    allowed=torch.where(torch.arange(len(u),device=dev)%5!=0)[0]
    with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            ids=[pool[torch.randint(len(pool),(n,),device=dev)] for pool,n in zip(pools,counts,strict=True)]
            observation=torch.cat([a[i] for a,i in zip(signals,ids,strict=True)])
            ground=torch.cat([a[i%len(a)] for a,i in zip(sources,ids,strict=True)])
            clean=u[allowed[torch.randint(len(allowed),(8,),device=dev)]]
            raw,error,_=training_terms(model,observation,ground)
            clean_error=shape_error(model.correction(*model.coordinates(clean))[0])
            objective=(error/(raw+.0004)).mean()+.15*(clean_error/.0004).mean()
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
            opt.step()
            sch.step()
            if step==1 or step%100==0:
                entry=dict(step=step,loss=float(objective.detach()),raw=float(raw.mean()),
                           restored=float(error.detach().mean()),identity=float(clean_error.detach().mean()))
                log.write(json.dumps(entry,allow_nan=False)+'\n')
                log.flush()
                print('LORA_COHERENT_DETAIL',seed,entry,flush=True)
    utility=fit_utility(model,l[0],u,sat,alt_u,held)
    write(out/'source_support.json',dict(training_pool=list(map(len,pools)),withheld_U=len(held),
          practical_exposures=STEPS*sum(counts),clean_exposures=STEPS*8,utility=utility))
    torch.save(model.state_dict(),out/'model.pt')
    write(out/'complete.json',record(c,seed))


def load_detail(root,c,seed,device):
    if root.name==DETAIL_DOSE_RUN:
        root=root.parent/DETAIL_RUN
    out=root/'G'/str(seed)
    if read(out/'complete.json')!=record(c,seed):
        raise ValueError('LoRa detail checkpoint source ancestry differs')
    model=PacketDetail(tracker_gate(root,c,device)).to(device)
    model.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True),strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def extra_correction(detail,x,old):
    ids=torch.where(detail.eligible(x)&~old)[0]
    if not len(ids):
        return ids,x[:0]
    wave=x[ids]
    delta=detail.proposal(wave)
    extra=detail.accepted(wave,delta)
    if not torch.isfinite(delta).all() or (strength_ratio(complex_iq(delta),complex_iq(wave))>CAP+1e-5).any():
        raise ValueError('Invalid high-coherence LoRa detail correction')
    return ids[extra],wave[extra]+delta[extra]


def scale_detail(raw,corrected,amount):
    return raw+amount*(corrected-raw)


@torch.no_grad()
def restore(base,detail,x):
    fixed,strength=base(x,with_gate=True)
    old=strength>0
    restored=x.clone()
    if old.any():
        restored[old]=protected_shapes(base,x[old],fixed[old])[0]
    selected,adjusted=extra_correction(detail,x,old)
    restored[selected]=adjusted
    active=old.clone()
    active[selected]=True
    return fixed,restored,old,active


def detail_smoke(rt):
    from .full_dura import RUN, engines
    root=Path(rt['run_root'])
    _,cache,c=handles(rt)
    seed=(CONFIRM_NEW_SEEDS if root.name==DETAIL_CONFIRM_RUN else DETAIL_SEEDS)[0]
    base,_=engines(root.parent/RUN,c,seed,'cpu')
    if root.name==DETAIL_DOSE_RUN:
        detail=load_detail(root,c,seed,'cpu')
        draw=load_draw(root.parent/DRAW_RUN,'L_s')
        x=tensor(draw[::16])
        extra,wave=extra_correction(detail,x,base.strength(x)>0)
        if not len(extra):
            raise ValueError('Frozen source detail smoke has no utility-gated proposals')
        for amount in DETAIL_DOSES.values():
            adjusted=scale_detail(x[extra],wave,amount)
            ratio=strength_ratio(complex_iq(adjusted-x[extra]),complex_iq(x[extra]))
            if (ratio>amount*CAP+1e-5).any() or torch.equal(adjusted,x[extra]):
                raise ValueError('Frozen detail dose did not satisfy its RMS bound')
        write(root/'smoke.json',dict(status='PASS',seed=seed,extra_active=len(extra),F_reads=0,
                                   TX_reads=0,target_reads=0,doses=DETAIL_DOSES))
        return
    model=PacketDetail(base.gate)
    l=np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)[0]
    old=np.load(cache/'U_s_practical.npy',mmap_mode='r',allow_pickle=False)[::32]
    a=load_draw(root.parent/DRAW_RUN,'L_s')
    x=tensor(a[::16])
    keep=torch.where(model.eligible(x))[0][:8]
    if len(keep)<8:
        raise ValueError('Real high-coherence source data are missing')
    x=x[keep]
    clean=tensor(l[::16][keep.tolist()])
    raw,error,_=training_terms(model,x,clean)
    error.mean().backward()
    params=[*model.common.parameters(),*model.individual.parameters()]
    grad=float(torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True))
    if not grad>0 or not torch.equal(raw,error):
        raise ValueError('Source detail gradient or initial identity failed')
    torch.optim.SGD(params,lr=.03).step()
    clean_sample=tensor(l[::61])
    fixed,updated,old_gate,new_gate=restore(base,model,clean_sample)
    delta=model.proposal(x)
    if (old_gate.any() or new_gate.any() or not torch.equal(fixed,clean_sample)
            or not torch.equal(updated,clean_sample) or not torch.isfinite(delta).all()
            or not (delta!=0).any() or not (base.strength(tensor(old))>0).any()):
        raise ValueError('Real source detail bypass, bound, or protected control failed')
    write(root/'smoke.json',dict(status='PASS',seed=seed,clean_exact=len(clean_sample),
              high_coherence=len(x),grad=grad,F_reads=0,TX_reads=0,target_reads=0))


@torch.no_grad()
def source_wave_detail(rt,seed):
    from .full_dura import RUN, engines
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    dev='cuda:0'
    base,_=engines(root.parent/RUN,c,seed,dev)
    detail=load_detail(root,c,seed,dev)
    arr=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    clean=tensor(arr[0,::4],dev)
    refs=[detail.coordinates(y)[0] for y in clean.split(48)]
    rows={}
    for i,v in enumerate(VIEWS):
        total=high=extra=better=old=clean_changed=0
        raw_error=restored_error=0.
        for x,ref in zip(tensor(arr[i,::4],dev).split(48),refs,strict=True):
            _,output,baseline,active=restore(base,detail,x)
            high+=int(detail.eligible(x).sum())
            extra_mask=active&~baseline
            a=shape_error(detail.coordinates(x)[0]-ref)
            b=shape_error(detail.coordinates(output)[0]-ref)
            extra+=int(extra_mask.sum())
            better+=int((b[extra_mask]<a[extra_mask]).sum())
            raw_error+=float(a[extra_mask].sum())
            restored_error+=float(b[extra_mask].sum())
            old+=int(baseline.sum())
            clean_changed+=int((output!=x).flatten(1).any(1).sum())
            total+=len(x)
        rows[v]=dict(n=total,high_coherence=high,existing_active=old,extra_active=extra,
                     extra_improved=better,extra_raw_error=raw_error,extra_restored_error=restored_error,
                     IQ_changed=clean_changed)
    write(root/'source_wave'/f'{seed}.json',dict(seed=seed,records=rows,target_reads=0,TX_reads=0,F_reads=0))


@torch.no_grad()
def predict_detail(root,target,c,seed,reader,device):
    from .full_dura import PROJECTED_MODES, RUN, STACK_RUN, engines
    dose=root.name==DETAIL_DOSE_RUN
    stack=root.parent/(DETAIL_RUN if dose else STACK_RUN)
    if read(stack/'state.json')['status']!='PREDICTIONS_FROZEN':
        raise ValueError('The frozen protected DURA controls are missing')
    old=stack/'predictions'/str(seed)
    old_receipt=read(old/'complete.json')
    controls=old_receipt.pop('activation')
    expected=dict(status='PREDICTIONS_FROZEN',seed=seed,target_truth_read=False,receivers=list(TARGET),
                  views=list(VIEWS),per_receiver=LIMIT)
    if old_receipt!=dict(expected,modes=list(DETAIL_MODES if dose else PROJECTED_MODES)):
        raise ValueError('The frozen Tri-set and DURA controls do not match')
    base,_=engines(root.parent/RUN,c,seed,device)
    detail=load_detail(root,c,seed,device)
    out=root/'predictions'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    records={}
    old_modes=DETAIL_MODES if dose else DETAIL_MODES[:-1]
    amounts={} if dose else {'coherent_detail':1.}
    if dose or root.name==DETAIL_CONFIRM_RUN:
        amounts.update(DETAIL_DOSES)
    new_modes=tuple(amounts)
    for rx in TARGET:
        waves=np.load(target/f'{rx}.npy',mmap_mode='r',allow_pickle=False)
        if waves.shape!=(7,LIMIT,6,2,256) or waves.dtype!=np.float32:
            raise ValueError('High-coherence target IQ differs')
        for i,v in enumerate(VIEWS):
            arrays=[np.load(old/f'{rx}__{v}__{m}.npy',allow_pickle=False) for m in old_modes]
            if any(a.shape!=(LIMIT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=10 for a in arrays):
                raise ValueError('Invalid frozen protected DURA predictions')
            updated=[arrays[2].copy() for _ in new_modes]
            n_old=n_extra=0
            for step in range(0,LIMIT,48):
                x=tensor(waves[i,step:step+48],device)
                baseline=base.strength(x)>0
                extra,result=extra_correction(detail,x,baseline)
                if len(extra):
                    waves_to_read=[result if amount==1. else scale_detail(x[extra],result,amount)
                                   for amount in amounts.values()]
                    logits=reader(torch.cat(waves_to_read))[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('High-coherence DURA produced nonfinite logits')
                    ids=extra.cpu().numpy()+step
                    for a,labels in zip(updated,logits.argmax(1).chunk(len(updated)),strict=True):
                        a[ids]=labels.cpu().numpy()
                    n_extra+=len(ids)
                n_old+=int(baseline.sum())
            key=f'{rx}/{v}'
            if controls[key]['active']!=n_old:
                raise ValueError('The existing protected branch active count changed')
            if dose and controls[key]['detail_extra']!=n_extra:
                raise ValueError('Frozen dose comparison changed the detail gate')
            records[key]=dict(controls[key],detail_active=n_old+n_extra,detail_extra=n_extra)
            for m,a in zip((*old_modes,*new_modes),(*arrays,*updated),strict=True):
                np.save(out/f'{rx}__{v}__{m}.npy',a)
        print('LORA_DETAIL_DURA_PREDICTED',seed,rx,flush=True)
    write(out/'complete.json',dict(expected,modes=[*old_modes,*new_modes],activation=records))
