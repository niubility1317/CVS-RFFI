"""Estimate a received LoRa relative action and preserve the observed common mode."""

from math import log

import torch
from torch import nn

from .channel_blind_source import waveform_risk
from .dura_repair import complex_iq, real_iq
from .joint_spectrum import limit, split
from .joint_unroll_source import coordinates
from .lapped_posterior import other_mean
from .natural_detail import paired_inputs, transfer_risk, weak_difference
from .split_shrink import spectrum


def constrained_step(y, prior, action, variance, penalty, *, anchor):
    """Observation-weighted ridge update with an optional fixed chirp mean."""
    precision = action.abs().square()/variance + penalty
    estimate = (action.conj()*y/variance + penalty*prior)/precision
    edit = estimate-y
    if anchor:
        inverse = precision.reciprocal()
        correction = edit.sum(1,keepdim=True)/inverse.sum(1,keepdim=True)
        edit = edit-inverse*correction
    return edit


def head(in_channels, outputs):
    layers = nn.Sequential(nn.Conv1d(in_channels,16,7,padding=3),nn.SiLU(),
                           nn.Conv1d(16,24,5,padding=4,dilation=2),nn.SiLU(),
                           nn.Conv1d(24,16,3,padding=1),nn.SiLU(),nn.Conv1d(16,outputs,1))
    nn.init.normal_(layers[-1].weight,std=.01)
    nn.init.zeros_(layers[-1].bias)
    return layers


def iq_field(*values):
    return tuple(field for x in values for field in (x.real,x.imag))


class CommonAnchor(nn.Module):
    def __init__(self, anchor_common):
        super().__init__()
        self.anchor = anchor_common
        self.estimator = head(11,3)
        self.coarse = head(14,2)
        self.refine = head(18,2)
        frequency = torch.fft.fftshift(torch.fft.fftfreq(256))
        self.register_buffer('outer',frequency.abs()>.25,persistent=False)
        self.register_buffer('time',torch.linspace(-1.,1.,256),persistent=False)

    def features(self, network, fields):
        matrix = torch.stack(fields,dim=2).flatten(0,1)
        bf16 = matrix.is_cuda and self.training and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            result = network(matrix)
        return result.float().unflatten(0,fields[0].shape[:2]).tanh().unbind(2)

    def solve(self, y, prior, action, variance, rms, cap, *, anchor, penalty):
        edit = constrained_step(y,prior,action,variance,penalty,anchor=anchor)
        edit = edit-edit.mean(-1,keepdim=True)
        return y+limit(edit,y,rms,cap)

    def states(self, y, q, rms, *, anchor_common=None):
        center = y.mean(1,keepdim=True)
        private = y-center
        other = other_mean(y)
        quality = q.expand_as(y.real)
        sigma = spectrum(y)[...,self.outer].abs().square().median(-1,keepdim=True).values/log(2.)
        fields = (*iq_field(y,center.expand_as(y),other,private),quality,sigma.log1p().expand_as(quality),
                  self.time.expand_as(quality))
        amp,theta,noise = self.features(self.estimator,fields)
        action = torch.polar((.35*amp).exp(),.85*theta)
        variance = (.015+sigma)*(1.5*noise).exp()
        observed_fields = (*fields,action.real-1.,action.imag,variance.log().clamp(-7.,3.)/4.)
        strength = .10 + .55*(sigma/(.025+sigma)).sqrt()
        proposal = y + strength*torch.complex(*self.features(self.coarse,observed_fields))
        cap = .065 + .145*((sigma.mean(1,keepdim=True).sqrt()-.045)/.30).clamp(0.,1.)
        anchor = self.anchor if anchor_common is None else anchor_common
        first = self.solve(y,proposal,action,variance,rms,cap,anchor=anchor,penalty=4.)
        innovation = action.conj()*(y-action*first)
        refined = (*observed_fields,*iq_field(first,innovation))
        proposal = first + .65*strength*torch.complex(*self.features(self.refine,refined))
        final = self.solve(y,proposal,action,variance,rms,cap,anchor=anchor,penalty=3.)
        return first,final,action,variance

    def stages(self, y, q, rms, *, anchor_common=None):
        first,final,action,variance = self.states(y,q,rms,anchor_common=anchor_common)
        common = spectrum(y.mean(1,keepdim=True))[...,self.outer].abs().square().median(-1).values[:,0]/log(2.)
        info = torch.stack((common,variance.mean((1,2)),q.mean((1,2)),(action-1.).abs().mean((1,2))),dim=1)
        return first,final,info


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean,angle,_,level = coordinates(tracker,iq)
    clean_frame = angle * level
    source = complex_iq(iq)
    detail = weak_difference(clean)
    extra = detail*clean_frame
    draws = paired_inputs(source,recipe.perturb,rng,extra,with_identity=with_identity)
    y,phase,q,level = coordinates(tracker,real_iq(draws.flatten(0,1)))
    frame = phase * level
    ref = torch.stack((source,source,source+extra),dim=1).flatten(0,1)/frame
    coarse,fine,action,variance = model.states(y,q,level)
    forward_error = (action*ref-y).abs().square()
    forward_fit = forward_error.log1p().mean()
    variance_fit = (forward_error.detach()/variance+variance.log()).mean()
    restored = (fine*frame).unflatten(0,draws.shape[:2])
    consistency = ((restored[:,0]-restored[:,1])/clean_frame).abs().square().mean()
    difference = split((restored[:,2]-restored[:,0])/clean_frame)
    target_difference = split(detail)
    private = transfer_risk(difference[:,1:],target_difference[:,1:])
    values = torch.stack((waveform_risk(coarse,ref),waveform_risk(fine,ref),consistency,private,
                          forward_fit,variance_fit))
    first,second,pair = recipe.loss_weights(step)
    loss = first*values[0] + second*values[1] + pair*consistency + .30*private + .45*forward_fit + .025*variance_fit
    return loss,values
