"""F-free LoRa restoration ablation under an exact clean-pass constraint."""
import json
import math

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import seed_all
from .packet_data import CHIRPS
from .packet_models import GuardedPacketDura, SEED
from .refine_f import PARENT, source_contract


METHODS_G=('phase','chirp')
G_EPOCHS=10
BATCH_G=24


class ActiveDura(GuardedPacketDura):
    @torch.no_grad()
    def strength(self,x):
        q=self.reference.coherence(x.flatten(0,1)).reshape(len(x),len(CHIRPS)).median(dim=1).values
        # Reliable high-coherence packets and textureless low-coherence packets
        # bypass G. Only the intermediate observations receive residuals.
        return ((.94-q)/.24).clamp(0.,1.)*((q-.20)/.25).clamp(0.,1.)

    def forward(self,x,*,with_gate=False):
        strength=self.strength(x)
        mask=strength>0.
        output=x.clone()
        if bool(mask.any()):
            accepted=x[mask]
            frames=accepted.flatten(0,1)
            full=self.net.stage_outputs(frames)[0][-1].reshape_as(accepted)
            residual=full-accepted
            rms=lambda z:z.square().mean((1,2,3)).clamp_min(1e-8).sqrt()
            scale=(.25*strength[mask]*rms(accepted)/rms(residual)).clamp_max(1.)
            output[mask]=accepted+scale[:,None,None,None]*residual
        return (output,strength) if with_gate else output


@torch.no_grad()
def aligned_target(clean,obs,reference,mode):
    a=torch.complex(clean[:,:,0],clean[:,:,1])
    b=torch.complex(obs[:,:,0],obs[:,:,1])
    if mode=='phase':
        phase=(a.conj()*b).sum((1,2),keepdim=True)
    elif mode=='chirp':
        ideal=reference.chirp.to(a.dtype).conj()
        # Estimate phase slope in each dechirped segment. Restrict alignment
        # to the nearest small residual offset supported by the received tone.
        sa=torch.fft.fft(a*ideal,n=2048)
        sb=torch.fft.fft(b*ideal,n=2048)
        diff=(sb.abs().argmax(2)-sa.abs().argmax(2)+1024)%2048-1024
        bq=(sb.abs().amax(2)/256).square()/b.abs().square().mean(2).clamp_min(1e-8)
        valid=(bq>=.30)&(diff.abs()<=16)
        off=torch.where(valid,diff,0)
        angle=2*torch.pi*off[:,:,None]*torch.arange(256,device=a.device)/2048
        a=a*torch.polar(torch.ones_like(angle),angle)
        local=(a.conj()*b).sum(2,keepdim=True)
        packet=(a.conj()*b).sum((1,2),keepdim=True)
        phase=torch.where(valid[:,:,None],local,packet)
    else:
        raise ValueError('Unknown DURA training objective')
    unit=torch.where(phase.abs()>1e-8,phase/phase.abs().clamp_min(1e-8),torch.ones_like(phase))
    y=a*unit
    return torch.stack((y.real,y.imag),dim=2)


def load_g_parent(parent,c,device,model=None):
    folder=parent/'G'
    g=read(folder/'complete.json')
    if (parent.name!=PARENT or g['status']!='SOURCE_FROZEN' or g['epochs']!=15 or g['seed']!=SEED
            or g['source_contract']!=c or g['F_reads'] or g['TX_labels_read'] or g['target_read']):
        raise ValueError('Exact same-contract F-free previous G required')
    model=(model or GuardedPacketDura()).to(device)
    model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True),strict=True)
    return model


def train(source,cache,parent,out,mode):
    if mode not in METHODS_G:
        raise ValueError('Unknown DURA refinement row')
    c=source_contract(source)
    out.mkdir(parents=True,exist_ok=False)
    seed_all(SEED)
    dev='cuda:0'
    pairs=tensor(np.load(cache/'G_pairs.npy',mmap_mode='r',allow_pickle=False),dev)
    expected=c['counts']['L_s']+c['counts']['U_s']
    if pairs.shape!=(2,expected,len(CHIRPS),2,256):
        raise ValueError('Aligned L/U restoration pairs differ')
    model=load_g_parent(parent,c,dev,ActiveDura())
    with torch.no_grad():
        mask=torch.cat([model.strength(row)>0 for row in pairs[1].split(64)])
        kept=torch.where(mask)[0]
    if len(kept)<BATCH_G:
        raise ValueError('Insufficient source-only recoverable paired IQ')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=G_EPOCHS*math.ceil(len(kept)/BATCH_G),eta_min=1e-5)
    with (out/'epochs.jsonl').open('x',encoding='utf8') as log:
        for epoch in range(1,G_EPOCHS+1):
            sums=torch.zeros(3,device=dev)
            model.train()
            for sel in torch.randperm(len(kept),device=dev).split(BATCH_G):
                y,x=pairs[:,kept[sel]]
                target=aligned_target(y,x,model.reference,mode)
                restored=model(x)
                wave=F_mse(restored,target)
                base=F_mse(x,target)
                limit=F_mse(restored,x)
                opt.zero_grad(set_to_none=True)
                (wave+.15*limit).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step();sch.step()
                sums+=torch.stack((wave.detach(),base,limit.detach()))*len(sel)
            record=dict(epoch=epoch,mode=mode,active=len(kept),total=expected,
                        reconstruction=(sums/len(kept)).tolist())
            print('REFINE_G_EPOCH',mode,json.dumps(record,allow_nan=False),flush=True)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
    torch.save(model.state_dict(),out/'model.pt')
    write(out/'complete.json',dict(status='SOURCE_FROZEN',mode=mode,epochs=G_EPOCHS,source_contract=c,
            seed=SEED,F_reads=False,TX_labels_read=False,target_read=False))


def F_mse(x,y):
    return (x-y).square().mean()
