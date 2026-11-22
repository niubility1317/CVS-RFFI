"""Same-contract LoRa IQ teacher and two-stage protected DURA student."""
import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.g_lineage import skeleton
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_r44_pruned_refine.models import spectral_error
from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
from .dura_gate import RiskDura
from .dura_repair import complex_iq, power
from .full_dura_core import BATCH, iq_arrays, protected_shapes, read_gate, receiver_loss
from .full_restore_draw import RUN as DRAW_RUN, load_draw
from .full_supervised import SEEDS, handles
from .refine_g import aligned_target


STEPS=2400


def record(c,seed,stage):
    return dict(status='SOURCE_FROZEN',source_contract=c,seed=seed,stage=stage,steps=STEPS,
                draw_run=DRAW_RUN,teacher_steps=STEPS if stage=='student' else 0,
                F_reads=0,TX_reads=0,target_reads=0)


def wrap(gate,device,model):
    restorer=RiskDura(gate,False)
    restorer.net=model
    return restorer.to(device)


def load_stage(root,c,seed,gate,device,stage):
    folder=root/'G'/stage/str(seed)
    if read(folder/'complete.json')!=record(c,seed,stage):
        raise ValueError('The LoRa stage checkpoint has a different source ancestry')
    model=skeleton('r65' if stage=='teacher' else 'r69')
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
    return wrap(gate,device,model).eval().requires_grad_(False)


def alternate_arrays(root,device):
    return [tensor(load_draw(root.parent/DRAW_RUN,role),device) for role in ('L_s','U_s')]


def corrected(model,x,proposal,strength):
    return protected_shapes(model,x,model.bounded(x,proposal,strength))[0]


def stage_loss(model,teacher,x,clean,identity,weight):
    target=aligned_target(clean,x,model.reference,'chirp')
    all_inputs=torch.cat((x,identity))
    raw=model.net.stage_outputs(all_inputs.flatten(0,1))[0]
    raw=[a.reshape_as(all_inputs) for a in raw]
    strength=model.strength(x)
    strength=torch.where(strength>0,strength,.60)
    restored=corrected(model,x,raw[-1][:len(x)],strength)
    aligned=receiver_loss(complex_iq(restored-target))
    identity_loss=receiver_loss(complex_iq(raw[-1][len(x):]-identity))
    drift=power(complex_iq(restored-x))
    paired=torch.cat((aligned+.025*drift,identity_loss)).mean()
    spectrum=spectral_error(restored.flatten(0,1),target.flatten(0,1)).mean()
    fidelity=paired.new_zeros(())
    if teacher is not None:
        with torch.no_grad():
            outputs=teacher.net.stage_outputs(x.flatten(0,1))[0]
            expected=[corrected(teacher,x,a.reshape_as(x),strength) for a in outputs[1:]]
        proposed=[corrected(model,x,raw[0][:len(x)],strength),restored]
        fidelity=torch.stack([receiver_loss(complex_iq(a-b)).mean() for a,b in
                              zip(proposed,expected,strict=True)]).mean()
    loss=paired+.05*spectrum+weight*fidelity
    return loss,dict(aligned=aligned.mean(),raw=receiver_loss(complex_iq(x-target)).mean(),
                    identity=identity_loss.mean(),spectrum=spectrum,teacher=fidelity)


def train_stage(rt,seed,stage):
    from .full_dura import RUN as ORIGINAL_G
    if seed not in SEEDS or stage not in ('teacher','student'):
        raise ValueError('Unregistered LoRa restoration stage')
    _,cache,c=handles(rt)
    root=Path(rt['run_root'])
    dev='cuda:0'
    torch.manual_seed(seed)
    rng=torch.Generator(device=dev).manual_seed(seed)
    gate=read_gate(root.parent/ORIGINAL_G/'G/gate',c,dev)
    gate.requires_grad_(False)
    teacher=load_stage(root,c,seed,gate,dev,'teacher') if stage=='student' else None
    backbone=StageDistilled(teacher.net) if teacher is not None else skeleton('r65')
    model=wrap(gate,dev,backbone.requires_grad_(True))
    model.train()
    l,u,sat=iq_arrays(cache,dev)
    alt_l,alt_u=alternate_arrays(root,dev)
    hold=torch.arange(len(u),device=dev)%5==0
    available_u=torch.where(~hold)[0]
    original_l=torch.where(torch.cat([model.strength(x)>0 for x in l[1:].flatten(0,1).split(192)]))[0]
    original_u=torch.where(torch.cat([model.strength(x)>0 for x in sat.split(192)])&~hold)[0]
    if min(len(original_l),len(original_u))<64:
        raise ValueError('Missing comparable original practical source support')
    clean_l=torch.arange(len(l[0]),device=dev)
    seen_l=torch.zeros(len(alt_l),device=dev,dtype=torch.bool)
    seen_u=torch.zeros(len(alt_u),device=dev,dtype=torch.bool)
    out=root/'G'/stage/str(seed)
    out.mkdir(parents=True,exist_ok=False)
    opt=torch.optim.AdamW(model.net.parameters(),lr=2e-4 if teacher is None else 1e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS,eta_min=2e-5)

    def draw(pool,n):
        return pool[torch.randint(len(pool),(n,),device=dev,generator=rng)]

    with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            j,i=draw(original_l,4),draw(original_u,4)
            a,b=draw(clean_l,6),draw(available_u,6)
            ji=j%len(l[0])
            clean=torch.cat((l[0,ji],u[i],l[0,a],u[b]))
            x=torch.cat((l[j//len(l[0])+1,ji],sat[i],alt_l[a],alt_u[b]))
            identity=torch.cat((l[0,draw(clean_l,2)],u[draw(available_u,2)]))
            if len(x)+len(identity)!=BATCH:
                raise ValueError('Unexpected teacher/student exposure count')
            seen_l[a]=True
            seen_u[b]=True
            weight=1.-.9*(step-1)/max(1,STEPS-1)
            opt.zero_grad(set_to_none=True)
            loss,terms=stage_loss(model,teacher,x,clean,identity,weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            sch.step()
            if step in (1,STEPS) or step%100==0:
                entry=dict(step=step,stage=stage,loss=float(loss.detach()),
                           **{k:float(v.detach()) for k,v in terms.items()})
                log.write(json.dumps(entry,allow_nan=False)+'\n')
                log.flush()
                print('LORA_STAGE_DURA',seed,entry,flush=True)
    write(out/'source_support.json',dict(original_L=len(original_l),original_U=len(original_u),
          alternate_L=int(seen_l.sum()),alternate_U=int(seen_u.sum()),held_U=int(hold.sum()),
          source_examples=STEPS*BATCH,active_practical_per_step=20,clean_per_step=4))
    torch.save(model.net.state_dict(),out/'model.pt')
    write(out/'complete.json',record(c,seed,stage))
