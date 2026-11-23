"""Joint WiSig time alignment and complex restoration from a single crop."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_rff_physics.known_excitation import template
from .wisig_timing import NOMINAL, PERIOD, long_template
from .wisig_uncertain_prior import as_complex, normalized_error, perturb


SAMPLES = 256
RADIUS = 40


def cnn(inputs, hidden, outputs):
    layers = nn.Sequential(nn.Conv1d(inputs,hidden,7,padding=3),nn.SiLU(),
                           nn.Conv1d(hidden,hidden,5,padding=4,dilation=2),nn.SiLU(),
                           nn.Conv1d(hidden,hidden,5,padding=2),nn.SiLU(),nn.Conv1d(hidden,outputs,1))
    nn.init.normal_(layers[-1].weight,std=.01)
    nn.init.zeros_(layers[-1].bias)
    return layers


def iq_fields(*parts):
    return tuple(field for value in parts for field in (value.real,value.imag))


class AnchoredWindow(nn.Module):
    def __init__(self, known_waveform):
        super().__init__()
        self.known_waveform = known_waveform
        self.timing = cnn(6,16,1)
        self.coarse = cnn(8,24,2)
        self.refine = cnn(10,24,2)
        time = np.arange(SAMPLES)
        shifts = np.arange(-RADIUS,RADIUS+1)
        indices = time[None]+shifts[:,None]
        boundary = NOMINAL+shifts
        relative = time[None]-boundary[:,None]
        long_view = relative>=0
        short_view = (time>=50)&(time<140)
        short_basis = template(relative)*short_view
        long_basis = long_template(relative)*long_view
        canonical = np.where(time<NOMINAL,template(time-NOMINAL),long_template(time-NOMINAL))
        long_extension = long_template(np.arange(80))
        before = boundary[:,None]-np.arange(PERIOD,0,-1)-PERIOD
        after = boundary[:,None]+np.arange(16)-PERIOD
        for name,value in dict(time=time,shifts=shifts,indices=indices,before=before,after=after,
                               short_basis=short_basis,long_basis=long_basis,canonical=canonical,long_extension=long_extension,
                               long_view=long_view).items():
            data = torch.tensor(value.tolist())
            if data.is_complex():
                data = data.to(torch.complex64)
            self.register_buffer(name,data,persistent=False)

    @staticmethod
    def oscillator(received, time):
        a,b = received[...,48:128], received[...,68:148]
        cross = (b*a.conj()).sum(-1)
        omega = cross.angle()/PERIOD
        periodic_noise = .5*(b-a*torch.polar(torch.ones_like(omega[:,None]),PERIOD*omega[:,None])).abs().square().mean(-1)
        rot = torch.polar(torch.ones_like(received.real),omega[:,None]*time)
        return rot, omega, periodic_noise

    @staticmethod
    def fit_basis(value, basis, visibility):
        energy = basis.abs().square().sum(-1).clamp_min(1e-6)
        correlation = value@basis.conj().T
        received_power = value.abs().square()@visibility.to(value.real.dtype).T
        coherence = (correlation.abs().square()/(energy*received_power.clamp_min(1e-6))).clamp(0.,1.)
        return correlation/energy, coherence

    def analyze(self, received):
        level = received.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
        raw = received/level
        rot,omega,noise = self.oscillator(raw,self.time)
        synced = raw*rot.conj()
        d = (synced[:,PERIOD:]-synced[:,:-PERIOD]).abs().square()
        energy = synced[:,PERIOD:].abs().square()+synced[:,:-PERIOD].abs().square()
        pre = d[:,self.before].sum(-1)/energy[:,self.before].sum(-1).clamp_min(1e-6)
        post = d[:,self.after].sum(-1)/energy[:,self.after].sum(-1).clamp_min(1e-6)
        jump = ((post+.05)/(pre+.05)).log()
        short_gain,short = self.fit_basis(synced,self.short_basis,self.short_basis!=0)
        long_gain,long = self.fit_basis(synced,self.long_basis,self.long_view)
        position = self.shifts.to(raw.real.dtype).expand_as(jump)/RADIUS
        if not self.known_waveform:
            short,long = torch.zeros_like(short),torch.zeros_like(long)
        fields = torch.stack((pre,post,jump,short,long,position),dim=1)
        logits = self.timing(fields).squeeze(1) + .8*jump + 2.*long + .3*short - .15*position.square()
        return raw,level,omega,noise,logits,short_gain,long_gain

    def select(self, raw, omega, logits, short_gain, long_gain):
        index = logits.argmax(-1)
        position = torch.arange(len(raw),device=raw.device)
        delta = self.shifts[index]
        margin = logits[position,index] - logits[:,RADIUS]
        if self.training:
            posterior = (logits/.5).softmax(-1)
            hard = F.one_hot(index,len(self.shifts)).to(posterior.dtype)
            weight = (hard+posterior-posterior.detach()).to(raw.dtype)
            banks, support, guide = self.moves(raw,omega,self.indices,short_gain,long_gain)
            aligned = (banks*weight[...,None]).sum(1)
            hint = (guide*weight[...,None]).sum(1)
            support = (support.to(raw.dtype)*weight[...,None]).sum(1).real
        else:
            idx = self.indices[index,None]
            aligned,support,hint = self.moves(raw,omega,idx,short_gain[position,index,None],long_gain[position,index,None])
            aligned,support,hint = aligned[:,0],support[:,0].to(raw.real.dtype),hint[:,0]
        return aligned,support,hint,delta,margin

    def moves(self, raw, omega, indices, short_gain, long_gain):
        observed_index = indices.expand(len(raw),-1,-1)
        valid = (observed_index>=0)&(observed_index<SAMPLES)
        directly_seen = raw.gather(-1,observed_index.clamp(0,SAMPLES-1).flatten(1)).unflatten(-1,observed_index.shape[1:])
        phase = torch.polar(torch.ones_like(directly_seen.real),omega[:,None,None]*observed_index)
        gain = torch.where(self.time<NOMINAL,short_gain[...,None],long_gain[...,None])
        predicted = phase*gain*self.canonical
        # Timing by itself preserves every genuinely observed overlap sample.
        return torch.where(valid,directly_seen,predicted),valid,predicted

    def restore(self, analysis):
        raw,level,omega,noise,logits,a,b = analysis
        aligned,support,guide,delta,margin = self.select(raw,omega,logits,a,b)
        available = noise[:,None].expand_as(support)
        trust = .12+.68*available/(.06+available)
        fields = (*iq_fields(aligned,guide,aligned-guide),support,available)
        mixed = self.training and raw.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            coarse_delta = self.coarse(torch.stack(fields,dim=1)).float()
        coarse = aligned + .45*trust*as_complex(coarse_delta.tanh())
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            fine_delta = self.refine(torch.stack((*fields,*iq_fields(coarse)),dim=1)).float()
        fine = coarse + .35*trust*as_complex(fine_delta.tanh())
        return (aligned*level,coarse*level,fine*level),delta,margin,noise

    def forward(self, received):
        return self.restore(self.analyze(received))


def generic_crop(source, rng, model):
    """Use public continuations and reflected boundaries in independent augmentations."""
    z = perturb(source[:,None],rng)[:,0]
    device = z.device
    batch = len(z)
    shift = torch.randint(-RADIUS,RADIUS+1,(batch,1),device=device,generator=rng)
    shift = shift*(torch.rand((batch,1),device=device,generator=rng)>.25)
    index = model.time[None]+shift
    reflected = z.gather(-1,torch.where(index<0,-index,torch.minimum(index,2*(SAMPLES-1)-index)))
    rot,omega,_ = model.oscillator(z,model.time)
    synced = z*rot.conj()
    short,long = model.canonical[:NOMINAL],model.canonical[NOMINAL:]
    early = (synced[:,50:140]*short[50:140].conj()).sum(-1)/short[50:140].abs().square().sum().clamp_min(1e-6)
    late = (synced[:,NOMINAL:]*long.conj()).sum(-1)/long.abs().square().sum().clamp_min(1e-6)
    prefix = short[index.remainder(PERIOD)]*early[:,None]
    # Only the unobserved extension is analytic. The in-window records are genuine IQ.
    long_index = NOMINAL+(index-NOMINAL).remainder(80)
    extra_basis = model.long_extension[long_index-NOMINAL]
    suffix = extra_basis*late[:,None]
    extrapolated = torch.where(index<0,prefix,suffix)*torch.polar(torch.ones_like(z.real),omega[:,None]*index)
    use = ((index<0)|(index>=SAMPLES)) & (torch.rand((batch,1),device=device,generator=rng)>.35)
    return torch.where(use,extrapolated,reflected)


def objective(model, source, rng, *, identity):
    views = torch.stack((generic_crop(source,rng,model),generic_crop(source,rng,model)),dim=1)
    if identity:
        views[-3:] = source[-3:,None]
    target = source[:,None].expand_as(views).flatten(0,1)
    (timed,coarse,fine),_,_,_ = model(views.flatten(0,1))
    paired = fine.unflatten(0,views.shape[:2])
    repeat = ((paired[:,0]-paired[:,1]).abs().square().mean(-1)
              /source.abs().square().mean(-1).clamp_min(.01)).mean()
    spectrum = torch.fft.fft(target,norm='ortho')
    error = torch.fft.fft(fine-target,norm='ortho').abs().square()
    relative = (error/(.02+spectrum.abs().square())).log1p().mean()
    costs = torch.stack((normalized_error(timed,target),normalized_error(coarse,target),normalized_error(fine,target),repeat,relative))
    return (costs*costs.new_tensor((.25,.40,1.,.10,.12))).sum(),costs
