"""Adapt the original R65 teacher and two-stage R69 to LoRa carrier coordinates."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import skeleton
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_r44_pruned_refine.models import spectral_error
from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
from .data import TARGET, VIEWS
from .dura_gate import ImpairmentGate, fit_gate
from .dura_repair import complex_iq, no_tone, real_iq
from .full_dura_detail import CAP, DETAIL_CONFIRM_RUN, DETAIL_DOSE_MODES, DETAIL_SEEDS, source_inputs, strength_ratio
from .full_packet_data import LIMIT
from .full_restore_draw import RUN as DRAW_RUN, load_draw
from .full_supervised import handles


R69_LORA_RUN='20261122-phase1-lora-dura-r69-aligned-m2-r01'
R69_MODES=('direct','fixed','protect_shape','coherent_half','r69_aligned','r69_aligned_half')
R69_SEEDS=DETAIL_SEEDS
R69_STEPS=2400
COUNTS=(6,6,4,4)
CLEAN=4


def carrier(gate,x):
    w,rot,_,_,_=gate.tracker.track(x)
    rms=w.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
    return w/rms,rot,rms


def canonical_stages(model,y):
    wave=real_iq(y)
    stages=model.stage_outputs(wave.flatten(0,1))[0]
    return [complex_iq(z.reshape_as(wave)) for z in stages]


def bounded_shape(y,proposal,rms):
    change=no_tone(proposal-y)
    ratio=strength_ratio(change*rms,y*rms)
    return change*(CAP/ratio).clamp_max(1.)[:,None,None]


def phase_wave_error(d):
    return d.real.square().mean((-2,-1))+2.*d.imag.square().mean((-2,-1))


def relative_error(y,clean,error):
    amplitude=error.real.square().mean((-2,-1))/(no_tone(y-clean).real.square().mean((-2,-1))+.001)
    phase=error.imag.square().mean((-2,-1))/(no_tone(y-clean).imag.square().mean((-2,-1))+.001)
    return amplitude+1.5*phase


def training_loss(model,teacher,y,clean,identity,rms,weight):
    stages=canonical_stages(model,torch.cat((y,identity)))
    proposed=[bounded_shape(y,s[:len(y)],rms) for s in stages]
    error=no_tone(y-clean)+proposed[-1]
    final=relative_error(y,clean,error).mean()
    per_stage=relative_error(y,clean,no_tone(y-clean)+proposed[0]).mean()
    id_loss=phase_wave_error(stages[-1][len(y):]-identity).mean()
    aligned=y+proposed[-1]
    aligned_clean=y-no_tone(y-clean)
    spectrum=spectral_error(real_iq(aligned).flatten(0,1),real_iq(aligned_clean).flatten(0,1)).mean()
    distill=final.new_zeros(())
    if teacher is not None:
        with torch.no_grad():
            expected=[bounded_shape(y,z,rms) for z in canonical_stages(teacher,y)[1:]]
        distill=torch.stack([relative_error(y,clean,a-b).mean() for a,b in
                             zip(proposed,expected,strict=True)]).mean()
    loss=final+.20*per_stage+15.*id_loss+.05*spectrum+weight*distill
    return loss,dict(raw_error=phase_wave_error(no_tone(y-clean)).mean(),
         recovered_error=phase_wave_error(error).mean(),identity=id_loss,spectrum=spectrum,distill=distill)


def evidence(c,seed,stage):
    return dict(status='SOURCE_FROZEN',source_contract=c,seed=seed,stage=stage,steps=R69_STEPS,
                coordinate='normalized_dechirped',cap=CAP,draw_run=DRAW_RUN,F_reads=0,TX_reads=0,target_reads=0)


def load_model(root,c,seed,stage,device):
    home=root/'G'/stage/str(seed)
    if read(home/'complete.json')!=evidence(c,seed,stage):
        raise ValueError('Wrong R69 LoRa source ancestry or stage')
    model=skeleton('r65' if stage=='teacher' else 'r69').to(device)
    model.load_state_dict(torch.load(home/'model.pt',map_location=device,weights_only=True),strict=True)
    return model.eval().requires_grad_(False)


def train_r69(rt,seed,stage):
    if seed not in R69_SEEDS or stage not in ('teacher','student'):
        raise ValueError('Unregistered original DURA LoRa row')
    root=Path(rt['run_root'])
    dev='cuda:0'
    torch.manual_seed(seed)
    c,(l,u,sat),(alt_l,alt_u)=source_inputs(rt,dev)
    tracker=ImpairmentGate().to(dev).eval().requires_grad_(False)
    teacher=load_model(root,c,seed,'teacher',dev) if stage=='student' else None
    model=(StageDistilled(teacher) if teacher is not None else skeleton('r65')).to(dev)
    model.train().requires_grad_(True)
    out=root/'G'/stage/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    inputs=(l[1:].flatten(0,1),sat,alt_l,alt_u)
    targets=(l[0],u,l[0],u)
    held=torch.arange(len(u),device=dev)%5==0
    u_pool=torch.where(~held)[0]
    pools=(torch.arange(len(inputs[0]),device=dev),u_pool,torch.arange(len(alt_l),device=dev),u_pool)
    seen=[torch.zeros(len(a),device=dev,dtype=torch.bool) for a in inputs]
    opt=torch.optim.AdamW(model.parameters(),lr=.0002 if teacher is None else .0001,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,R69_STEPS,eta_min=.00002)
    with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,R69_STEPS+1):
            ids=[p[torch.randint(len(p),(n,),device=dev)] for p,n in zip(pools,COUNTS,strict=True)]
            for flags,i in zip(seen,ids,strict=True):
                flags[i]=True
            received=torch.cat([a[i] for a,i in zip(inputs,ids,strict=True)])
            clean=torch.cat([a[i%len(a)] for a,i in zip(targets,ids,strict=True)])
            id_wave=u[u_pool[torch.randint(len(u_pool),(CLEAN,),device=dev)]]
            y,_,rms=carrier(tracker,received)
            clean,id_wave=(carrier(tracker,x)[0] for x in (clean,id_wave))
            weight=1.-.9*(step-1)/(R69_STEPS-1)
            loss,terms=training_loss(model,teacher,y,clean,id_wave,rms,weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            schedule.step()
            if step==1 or step%100==0:
                entry=dict(step=step,stage=stage,loss=float(loss.detach()),
                           **{k:float(v.detach()) for k,v in terms.items()})
                log.write(json.dumps(entry,allow_nan=False)+'\n')
                log.flush()
                print('LORA_R69_ALIGNED',seed,entry,flush=True)
    torch.save(model.state_dict(),out/'model.pt')
    write(out/'source_support.json',dict(stage=stage,seen=[int(v.sum()) for v in seen],
          practical_exposures=R69_STEPS*sum(COUNTS),clean_exposures=R69_STEPS*CLEAN,held_U=int(held.sum())))
    write(out/'complete.json',evidence(c,seed,stage))


@torch.no_grad()
def source_features(gate,inputs):
    return torch.cat([gate.features(x) for x in inputs.split(192)])


def utility(rt,seed):
    root=Path(rt['run_root'])
    c,(l,u,sat),(_,alt_u)=source_inputs(rt,'cuda:0')
    dev=u.device
    model=load_model(root,c,seed,'student',dev)
    torch.manual_seed(seed)
    gate=ImpairmentGate().to(dev)
    held=torch.arange(0,len(u),5,device=dev)
    held_train,held_probe=held[::2],held[1::2]
    lclean=source_features(gate,l[0])
    uclean=source_features(gate,u)

    @torch.no_grad()
    def benefit(a,ids):
        out=[]
        for index in ids.split(96):
            x=a[index]
            b,_,rms=carrier(gate,x)
            z,_,_=carrier(gate,u[index])
            e=no_tone(b-z)
            correction=bounded_shape(b,canonical_stages(model,b)[-1],rms)
            before=phase_wave_error(e)
            after=phase_wave_error(e+.5*correction)
            out.append((before-after)/before.clamp_min(.0001))
        return torch.cat(out)

    signals=(sat,alt_u)
    candidates=[source_features(gate,x[held_train])[benefit(x,held_train)>.05] for x in signals]
    positives=torch.cat(candidates)
    if len(positives)<96:
        raise ValueError('Too few source practical pairs benefit from R69')
    trace=fit_gate(gate,torch.cat((lclean,uclean[held_train])),positives,35,seed,uclean)
    checks=[]
    for x in signals:
        with torch.no_grad():
            observed=source_features(gate,x[held_probe])
            k=(observed[:,0]>.20)&(gate.logits(observed)>gate.clean_boundary)
            improvements=benefit(x,held_probe)[k]
        checks.append(dict(n=len(k),activated=int(k.sum()),beneficial=int((improvements>0).sum()),
                           mean_improvement=float(improvements.mean()) if len(improvements) else 0.))
    home=root/'G'/'gate'/str(seed)
    home.mkdir(parents=True,exist_ok=False)
    torch.save(gate.state_dict(),home/'model.pt')
    write(home/'training.json',dict(history=trace,positives=len(positives),held_U_fitting=len(held_train),
         held_U_probe=len(held_probe),checks=checks,clean_threshold=float(gate.clean_boundary)))
    write(home/'complete.json',evidence(c,seed,'gate'))


def load_utility(root,c,seed,device):
    home=root/'G'/'gate'/str(seed)
    if read(home/'complete.json')!=evidence(c,seed,'gate'):
        raise ValueError('Wrong R69 LoRa source utility ancestry')
    gate=ImpairmentGate().to(device)
    gate.load_state_dict(torch.load(home/'model.pt',map_location=device,weights_only=True),strict=True)
    return gate.eval().requires_grad_(False)


@torch.no_grad()
def restored(gate,model,x):
    y,rot,rms=carrier(gate,x)
    step=bounded_shape(y,canonical_stages(model,y)[-1],rms)
    delta=real_iq(step*rot*rms)
    if not torch.isfinite(delta).all() or (strength_ratio(complex_iq(delta),complex_iq(x))>CAP+1e-5).any():
        raise ValueError('R69 LoRa correction violates observed IQ bounds')
    return x+delta,x+.5*delta


def r69_smoke(rt):
    from .full_dura import RUN, engines
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    seed,=R69_SEEDS
    known,_=engines(root.parent/RUN,c,seed,'cpu')
    tracker=ImpairmentGate()
    a=np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)
    alternates=load_draw(root.parent/DRAW_RUN,'L_s')
    x,z=tensor(alternates[:3]),tensor(a[0,:3])
    y,_,rms=carrier(tracker,x)
    clean=carrier(tracker,z)[0]
    student=skeleton('r69')
    loss,_=training_loss(student,None,y,clean,clean,rms,0.)
    loss.backward()
    g=float(torch.nn.utils.clip_grad_norm_(student.parameters(),5.,error_if_nonfinite=True))
    modes=restored(tracker,student,x)
    if not g>0 or not all(q.shape==x.shape for q in modes) or not torch.isfinite(known.proposal(x)).all():
        raise ValueError('Real-IQ original DURA adaptation smoke failed')
    write(root/'smoke.json',dict(status='PASS',grad=g,real_n=len(x),F_reads=0,TX_reads=0,target_reads=0))


@torch.no_grad()
def r69_audit(rt,seed):
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    dev='cuda:0'
    gate=load_utility(root,c,seed,dev)
    model=load_model(root,c,seed,'student',dev)
    a=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    rows={}
    for j,v in enumerate(VIEWS):
        count=0
        sums=torch.zeros(3,device=dev)
        for raw,clean in zip(tensor(a[j,::12],dev).split(64),tensor(a[0,::12],dev).split(64),strict=True):
            k=gate.impairment(raw)>0
            count+=int(k.sum())
            if k.any():
                x,z=raw[k],clean[k]
                ground=no_tone(carrier(gate,z)[0])
                inputs=(x,*restored(gate,model,x))
                sums+=torch.stack([phase_wave_error(no_tone(carrier(gate,w)[0])-ground).sum()
                                   for w in inputs])
        rows[v]=dict(n=1000,active=count,phase_wave_error=(sums/max(count,1)).tolist())
    write(root/'source_wave'/f'{seed}.json',dict(seed=seed,records=rows,F_reads=0,TX_reads=0,target_reads=0))


@torch.no_grad()
def predict_r69(root,target,c,seed,reader,device):
    reference=root.parent/DETAIL_CONFIRM_RUN
    if read(reference/'state.json')['status']!='PREDICTIONS_FROZEN':
        raise ValueError('R69 LoRa comparison needs the frozen half-rate control')
    old=reference/'predictions'/str(seed)
    receipt=read(old/'complete.json')
    previous=receipt.pop('activation')
    expected=dict(status='PREDICTIONS_FROZEN',seed=seed,target_truth_read=False,receivers=list(TARGET),
                  views=list(VIEWS),per_receiver=LIMIT)
    if receipt!=dict(expected,modes=list(DETAIL_DOSE_MODES)):
        raise ValueError('Wrong existing R69 LoRa comparison controls')
    gate=load_utility(root,c,seed,device)
    model=load_model(root,c,seed,'student',device)
    out=root/'predictions'/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    records={}
    for rx in TARGET:
        arrays=np.load(target/f'{rx}.npy',mmap_mode='r',allow_pickle=False)
        if arrays.shape!=(len(VIEWS),LIMIT,6,2,256) or arrays.dtype!=np.float32:
            raise ValueError('R69 LoRa target IQ contract differs')
        for i,v in enumerate(VIEWS):
            controls=[np.load(old/f'{rx}__{v}__{m}.npy',allow_pickle=False) for m in R69_MODES[:-2]]
            if any(a.shape!=(LIMIT,) or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>=10 for a in controls):
                raise ValueError('Invalid matched R69 LoRa comparison controls')
            current=[controls[0].copy(),controls[0].copy()]
            n=0
            for step in range(0,LIMIT,48):
                x=tensor(arrays[i,step:step+48],device)
                k=gate.impairment(x)>0
                if k.any():
                    selected=np.asarray(torch.where(k)[0].tolist())+step
                    logits=reader(torch.cat(restored(gate,model,x[k])))[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError('R69 produced nonfinite class logits')
                    for values,labels in zip(current,logits.argmax(1).chunk(2),strict=True):
                        values[selected]=labels.cpu().numpy()
                    n+=len(selected)
            records[f'{rx}/{v}']=dict(previous[f'{rx}/{v}'],r69_active=n)
            for mode,values in zip(R69_MODES,(*controls,*current),strict=True):
                np.save(out/f'{rx}__{v}__{mode}.npy',values)
        print('LORA_R69_ALIGNED_PREDICTED',seed,rx,flush=True)
    write(out/'complete.json',dict(expected,modes=list(R69_MODES),activation=records))
