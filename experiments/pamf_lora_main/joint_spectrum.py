"""Couple shared and chirp-specific complex corrections from received LoRa IQ."""

from math import log

import torch
from torch import nn

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .split_shrink import spectrum, wave


def split(value):
    common = value.mean(1, keepdim=True)
    return torch.cat((common, value-common), dim=1)


def combine(parts):
    private = parts[:,1:]
    return parts[:,:1] + private - private.mean(1,keepdim=True)


def head(inputs):
    return nn.Sequential(nn.Conv1d(inputs,24,5,padding=2), nn.SiLU(),
                         nn.Conv1d(24,32,5,padding=4,dilation=2), nn.SiLU(),
                         nn.Conv1d(32,24,3,padding=1), nn.SiLU(), nn.Conv1d(24,2,1))


def limit(edit, observed, rms, cap):
    power = (edit*rms).abs().square().mean((1,2),keepdim=True)
    allowance = cap.square() * (observed*rms).abs().square().mean((1,2),keepdim=True)
    scale = (allowance/power.clamp_min(1e-12)).clamp(1e-12,1.).sqrt()
    return edit * scale


class JointSpectrum(nn.Module):
    def __init__(self, exchange):
        super().__init__()
        self.exchange = exchange
        self.coarse = head(10)
        self.refine = head(18)
        for network in (self.coarse,self.refine):
            nn.init.normal_(network[-1].weight, std=.008)
            nn.init.zeros_(network[-1].bias)
        f = torch.fft.fftshift(torch.fft.fftfreq(256))
        self.register_buffer('frequency',f,persistent=False)
        self.register_buffer('outer',f.abs()>.25,persistent=False)
        self.register_buffer('noncarrier',f!=0,persistent=False)
        self.register_buffer('which',torch.tensor([1.,0.,0.,0.,0.,0.,0.])[None,:,None],persistent=False)

    def predict(self, features, network):
        use_bf16 = self.training and features.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            result = network(features.flatten(0,1))
        real,imag = result.float().unflatten(0,features.shape[:2]).tanh().unbind(2)
        return torch.complex(real,imag)

    @staticmethod
    def neighbor(value):
        return .5 * (value.roll(2,dims=-1) + value.roll(-2,dims=-1))

    @staticmethod
    def exchange_parts(value, quality):
        private = value[:,1:]
        strength = private.abs()
        weights = (.2+.8*quality) * strength / (strength+.035)
        pooled = (private*weights).sum(1,keepdim=True)/weights.sum(1,keepdim=True).clamp_min(1e-6)
        return torch.cat((pooled,value[:,:1].expand_as(private)),dim=1)

    def stages(self, observed, quality, rms, *, exchange=None):
        source = spectrum(observed)
        parts = split(source)
        sigma = parts[...,self.outer].abs().square().median(-1,keepdim=True).values / log(2.)
        spread = parts[:,1:].abs().square().mean(1,keepdim=True)
        odd, even = (source[:,offset::2].mean(1,keepdim=True) for offset in (0,1))
        agreement = (odd*even.conj()).real.clamp_min(0.)
        discord = .5*(odd-even).abs().square()
        confidence = agreement / (.005 + agreement + discord)
        roles = torch.cat((quality.mean(1,keepdim=True),quality),dim=1)
        normalization = (.012 + sigma).sqrt()
        strength = (sigma/(.02+sigma)).sqrt()
        allowance = normalization * strength * self.noncarrier
        public_guard = 1. - .65 * self.which * confidence
        scaled = parts / normalization
        features = torch.stack((scaled.real.asinh(), scaled.imag.asinh(), scaled.abs().log1p(),
                                sigma.log1p().expand_as(parts.real), spread.log1p().expand_as(parts.real),
                                confidence.expand_as(parts.real), discord.log1p().expand_as(parts.real),
                                roles.expand_as(parts.real), self.frequency.expand_as(parts.real),
                                self.which.expand_as(parts.real)),dim=2)
        proposal = .85 * allowance * public_guard * self.predict(features,self.coarse)
        caps = .07 + .26 * ((sigma[:,1:].mean(1,keepdim=True).sqrt()-.035)/.25).clamp(0.,1.)
        first = limit(wave(combine(proposal)),observed,rms,caps)
        updated = parts + split(spectrum(first))
        # The instantaneous time-domain residual can be broadband even at a weak spectral bin.
        residual_level = split(-first).abs().square().mean(-1,keepdim=True).log1p().expand_as(parts.real)
        cross = self.exchange if exchange is None else exchange
        context = self.exchange_parts(updated,quality) if cross else self.neighbor(updated)
        context_delta = self.exchange_parts(updated-parts,quality) if cross else self.neighbor(updated-parts)
        local = updated / normalization
        context,context_delta = (x / normalization for x in (context,context_delta))
        extension = torch.stack((local.real.asinh(), local.imag.asinh(), local.abs().log1p(), residual_level,
                                 context.real.asinh(), context.imag.asinh(), context_delta.real.asinh(),
                                 context_delta.imag.asinh()),dim=2)
        additional = .65 * allowance * public_guard * self.predict(torch.cat((features,extension),dim=2),self.refine)
        fine = limit(first+wave(combine(additional)),observed,rms,caps)
        estimates = torch.stack((sigma[:,:1,0].mean(1),sigma[:,1:,0].mean(1),confidence.mean((1,2)),
                                 context.abs().mean((1,2))),dim=1)
        return observed+first, observed+fine, estimates


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean, phase, _, rms = coordinates(tracker, iq)
    clean_frame = phase * rms
    physical = complex_iq(iq)
    detail = weak_difference(clean)
    extra = detail * clean_frame
    views = paired_inputs(physical,recipe.perturb,rng,extra,with_identity=with_identity)
    observed,phase,q,rms = coordinates(tracker,real_iq(views.flatten(0,1)))
    basis = phase * rms
    target = torch.stack((physical,physical,physical+extra),dim=1).flatten(0,1) / basis
    coarse,fine,_ = model.stages(observed,q,rms)
    result = (fine*basis).unflatten(0,views.shape[:2])
    consistency = ((result[:,0]-result[:,1])/clean_frame).abs().square().mean()
    passage = split((result[:,2]-result[:,0])/clean_frame)
    shared, private = split(detail).split((1,detail.shape[1]),dim=1)
    common_risk = transfer_risk(passage[:,:1],shared)
    private_risk = transfer_risk(passage[:,1:],private)
    values = torch.stack((waveform_risk(coarse,target),waveform_risk(fine,target),consistency,common_risk,private_risk))
    first,second,pair = recipe.loss_weights(step)
    loss = first*values[0] + second*values[1] + pair*values[2] + .20*(common_risk+private_risk)
    return loss,values
