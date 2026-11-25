"""Recover LoRa residual fingerprints while preserving the observed chirp tone."""
import math

import numpy as np
import torch
from torch import nn

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import dura_model
from .physical import ChirpResponse
from .refine_g import ActiveDura, aligned_target

MODES=('projection','coherent')
CLEAN_PASS_Q=.82


def complex_iq(x):
    return torch.complex(x[:,:,0],x[:,:,1])


def real_iq(x):
    return torch.stack((x.real,x.imag),dim=2)


def no_tone(x):
    return x-x.mean(-1,keepdim=True)


def power(x):
    return x.abs().square().mean((-2,-1))


class PacketRestorer(nn.Module):
    """Keep carrier direction and fit a learned residual dose in two-stage DURA."""

    def __init__(self, mode, *, samples=256):
        super().__init__()
        self.mode=mode
        self.reference=ChirpResponse(fft_factor=8)
        if samples != 256:
            from .data import BANDWIDTH_HZ, WINDOW_START
            t = torch.arange(samples, dtype=torch.float64) / (samples * 1000) + WINDOW_START / 1e6
            self.reference.chirp = torch.exp(2j * torch.pi * BANDWIDTH_HZ * (t.square() / .002048 - t / 2)).to(torch.complex64)
            self.reference.fft_size = 8 * samples
        self.net=None if mode=='consensus' else dura_model()
        if self.net is not None:
            self.dose=nn.Sequential(nn.Linear(4,16),nn.SiLU(),nn.Linear(16,2))
            with torch.no_grad():
                self.dose[-1].weight.zero_()
                self.dose[-1].bias.copy_(torch.tensor([-.8,-.4]))

    @torch.no_grad()
    def track(self,x):
        z=complex_iq(x)
        w=z*self.reference.chirp.conj()
        size, n = self.reference.fft_size, z.shape[-1]
        peak,k=torch.fft.fft(w,n=size).abs().max(-1)
        coarse_q=(peak/n).square()/z.abs().square().mean(-1).clamp_min(1e-12)
        k=(k+size//2)%size-size//2
        t=torch.arange(n,device=x.device)
        angle=2*torch.pi*k[...,None]*t/size
        rot=self.reference.chirp*torch.polar(torch.ones_like(angle),angle)
        w=z*rot.conj()
        # Fine timing-slope alignment, clamped inside the FFT peak bin.
        lag=3*n//32
        phi=(w[...,lag:]*w[...,:-lag].conj()).mean(-1).angle()/lag
        angle=phi.clamp(-torch.pi/size,torch.pi/size)[...,None]*t
        rot=rot*torch.polar(torch.ones_like(angle),angle)
        w=z*rot.conj()
        q=(w.mean(-1).abs().square()/w.abs().square().mean(-1).clamp_min(1e-8)).clamp(0.,1.)
        dc=w.mean(-1,keepdim=True)
        phase=dc/ dc.abs().clamp_min(1e-8)
        # A zero tone may occur in noise-only data, which the gate bypasses.
        phase=torch.where(dc.abs()>1e-8,phase,torch.ones_like(phase))
        rot=rot*phase
        w=z*rot.conj()
        q_packet=q.median(1).values
        gate=((CLEAN_PASS_Q-q_packet)/.24).clamp(0.,1.)*((q_packet-.20)/.25).clamp(0.,1.)
        gate=torch.where(coarse_q.median(1).values>=CLEAN_PASS_Q,torch.zeros_like(gate),gate)
        return w,rot,q,q_packet,gate

    def forward(self,x,*,with_gate=False):
        w,rot,q,median,gate=self.track(x)
        result=x.clone()
        selected=gate>0
        if selected.any():
            y=x[selected]
            w,rot,q=w[selected],rot[selected],q[selected]
            strength=gate[selected]
            energy=power(w).clamp_min(1e-8).sqrt()
            shape=no_tone(w)
            amplitude=w.abs().square().mean(-1).clamp_min(1e-8).sqrt()
            sig=shape/amplitude[...,None]
            weights=q.square()[...,None]
            others=(sig*weights).sum(1,keepdim=True)-sig*weights
            others=others/(weights.sum(1,keepdim=True)-weights).clamp_min(1e-6)
            common=no_tone(others*amplitude[...,None]-shape)
            if self.net is None:
                proposed=common
                gain=torch.full_like(strength,.50)
            else:
                separation=power(common).sqrt()/energy
                context=torch.stack((median[selected],q.min(1).values,q.max(1).values,separation),1)
                gain,mix=self.dose(context).sigmoid().unbind(1)
                received=y.flatten(0,1)
                stages=self.net.stage_outputs(received)[0]
                candidate=complex_iq(stages[-1].reshape_as(y))
                proposed=no_tone(candidate*rot.conj()-w)
                if self.mode=='coherent':
                    # Both directions are normalized before mixing.
                    learned=proposed/power(proposed).clamp_min(1e-8).sqrt()[:,None,None]
                    repeated=common/power(common).clamp_min(1e-8).sqrt()[:,None,None]
                    proposed=learned*(1.-mix[:,None,None])+repeated*mix[:,None,None]
            norm=power(proposed).clamp_min(1e-8).sqrt()
            # Learning the dose after smooth direction normalization
            # retains a magnitude gradient even for a large raw residual.
            cap=.35*strength*energy*gain
            direction=proposed / (norm.square()+(.12*energy).square()).sqrt()[:,None,None]
            result[selected]=y+real_iq(direction*rot*cap[:,None,None])
        return (result,gate) if with_gate else result


def old_restorer(run, contract, device):
    g=run/'G'/'chirp'
    meta=read(g/'complete.json')
    if (meta['source_contract']!=contract or meta['status']!='SOURCE_FROZEN' or meta['mode']!='chirp'
            or meta['epochs']!=10 or meta['F_reads'] or meta['TX_labels_read'] or meta['target_read']):
        raise ValueError('Previous LoRa DURA does not match the paired source contract')
    m=ActiveDura().to(device).eval()
    m.load_state_dict(torch.load(g/'model.pt',map_location=device,weights_only=True),strict=True)
    return m


def losses(f,clean,x,restore):
    with torch.no_grad():
        target=aligned_target(clean,x,f.reference,'chirp')
        w,rot,_,_,_=f.track(x)
        residual_target=no_tone(complex_iq(target)*rot.conj()-w)
    correction=complex_iq(restore-x)*rot.conj()
    error=correction-residual_target
    ms=power(error).mean()
    common=error.mean(1).abs().square().mean()
    base=power(residual_target).mean()
    return ms,common,base,power(correction).mean()


def paired_training(cache,device):
    # Both arrays contain IQ only, no classifier output, TX, V or target.
    a=np.load(cache/'G_pairs.npy',mmap_mode='r',allow_pickle=False)
    labeled=np.load(cache/'L_s.npy',mmap_mode='r',allow_pickle=False)
    if a.shape!=(2,4160,6,2,256) or labeled.shape!=(7,1600,6,2,256):
        raise ValueError('Pilot aligned restoration pairs differ')
    clean=tensor(np.concatenate((np.tile(labeled[0],(6,1,1,1)),a[0,1600:])),device)
    practical=tensor(np.concatenate((labeled[1:].reshape(-1,6,2,256),a[1,1600:])),device)
    return clean,practical


def train(cache,previous,out,contract,mode,seed,epochs):
    if mode not in MODES:
        raise ValueError('Unknown DURA ablation arm')
    dev='cuda:0'
    torch.manual_seed(seed)
    model=PacketRestorer(mode).to(dev)
    model.net.load_state_dict(old_restorer(previous,contract,dev).net.state_dict(),strict=True)
    clean,x=paired_training(cache,dev)
    eligible=torch.cat([model.track(b)[-1]>0 for b in x.split(96)])
    available=torch.where(eligible)[0]
    if len(available)<128:
        raise ValueError('DURA needs enough recoverable aligned source IQ')
    out.mkdir(parents=True,exist_ok=False)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    steps=math.ceil(len(available)/24)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*steps,eta_min=2e-5)
    from json import dumps
    with (out/'epochs.jsonl').open('x',encoding='utf8') as log:
        for e in range(1,epochs+1):
            meter=torch.zeros(4,device=dev)
            model.train()
            for ids in available[torch.randperm(len(available),device=dev)].split(24):
                sample=x[ids]
                out_i=model(sample)
                m,c,b,d=losses(model,clean[ids],sample,out_i)
                opt.zero_grad(set_to_none=True)
                (m+.35*c+.025*d).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step()
                sch.step()
                meter+=torch.stack((m,c,b,d)).detach()*len(ids)
            row=dict(epoch=e,active=len(available),total=len(x),steps=e*steps,
                     tone_residual_mse=(meter/len(available)).tolist())
            line=dumps(row,allow_nan=False)
            log.write(line+'\n')
            log.flush()
            print('DURA_PACKET_REPAIR',mode,line,flush=True)
    torch.save(model.state_dict(),out/'model.pt')
    write(out/'complete.json',dict(status='SOURCE_FROZEN',mode=mode,seed=seed,epochs=epochs,
              source_contract=contract,F_reads=False,TX_labels_read=False,target_read=False))
