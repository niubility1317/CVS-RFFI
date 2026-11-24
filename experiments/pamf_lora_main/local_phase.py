"""Restore local LoRa phase in the observed amplitude and timing coordinates."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F

from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates
from .natural_detail import paired_inputs, transfer_risk
from .spectral_block_recovery import varied_observation


def rotation(angle):
    return torch.polar(torch.ones_like(angle), angle)


def curvature(wave):
    unit = wave/wave.abs().clamp_min(1e-6)
    return torch.stack([F.pad(unit[..., 2*lag:]*unit[..., :-2*lag]*unit[..., lag:-lag].conj().square(), (lag,lag))
                        for lag in (1,4,12)], dim=2)


def window_mean(x):
    return F.avg_pool1d(x, 33, stride=1, padding=16, count_include_pad=False)


def repeat_mean(value, weight):
    weighted = value*weight
    normalizer = weight.sum(1,keepdim=True)-weight
    return (weighted.sum(1,keepdim=True)-weighted)/normalizer.clamp_min(1e-6)


def read_observation(y, quality, high_band, *, use_action=True, use_noise=True):
    abs_y = y.abs()
    reliability = .1+.9*quality
    other = repeat_mean(y,reliability)
    error = (y-other).abs().square()
    w = reliability/(1.+error/(.02+window_mean(error)))
    reference = repeat_mean(y,w)
    power = window_mean(w*reference.abs().square())
    if use_action:
        numerator = w*y*reference.conj()
        fit = torch.complex(window_mean(numerator.real),window_mean(numerator.imag))
        mismatch = window_mean((y-reference).abs().square())
        trust = power/(.04+power+mismatch)
        action = 1.+trust*(fit-power)/(.12+power)
    else:
        action = torch.ones_like(y)
    repeated = action*reference
    disagreement = window_mean((y-repeated).abs().square())
    if use_noise:
        spec = torch.fft.fft(y,norm='ortho')
        noise = spec[..., high_band].abs().square().median(-1,keepdim=True).values/log(2.)
    else:
        noise = torch.full_like(quality,.025)
    confidence = quality*power/(.02+power+disagreement+noise)
    angles = curvature(y)
    common = repeat_mean(angles,w.unsqueeze(2))
    return repeated, action, noise, disagreement, confidence, abs_y, angles, common


def project(proposed, power, window, time, *, cap=.10, angle_cap=.35):
    weight = power*window
    mass = weight.sum(-1,keepdim=True)
    safe_mass = mass.clamp_min(1e-8)
    centered = time-(weight*time).sum(-1,keepdim=True)/safe_mass
    moment = (weight*centered.square()).sum(-1,keepdim=True)
    offset = (weight*proposed).sum(-1,keepdim=True)/safe_mass
    slope = (weight*proposed*centered).sum(-1,keepdim=True)/moment.clamp_min(1e-8)
    delta = (window*(proposed-offset-slope*centered)).masked_fill((mass<1e-8)|(moment<1e-8),0.)
    ratio = (power*delta.square()).sum((1,2),keepdim=True)/power.sum((1,2),keepdim=True).clamp_min(1e-8)
    dose = (cap*ratio.clamp_min(1e-8).rsqrt()).clamp_max(1.)
    maximum = (angle_cap/delta.abs().amax((1,2),keepdim=True).clamp_min(1e-8)).clamp_max(1.)
    return delta*torch.minimum(dose,maximum)


def phase_head(inputs):
    result = nn.Sequential(nn.Conv1d(inputs,32,5,padding=2),nn.SiLU(),
                           nn.Conv1d(32,32,5,padding=8,dilation=4),nn.SiLU(),
                           nn.Conv1d(32,1,1))
    nn.init.zeros_(result[-1].bias)
    nn.init.normal_(result[-1].weight,std=.01)
    return result


class LocalPhase(nn.Module):
    def __init__(self, edges):
        super().__init__()
        t = torch.arange(256)-127.5
        edge = .5-.5*torch.cos(torch.pi*((t.abs()-47.5)/16).clamp(0.,1.))
        self.register_buffer('window',edge if edges else edge.roll(128),persistent=False)
        self.register_buffer('time',t/127.5,persistent=False)
        f = torch.fft.fftfreq(256).abs()
        self.register_buffer('high_band',f>.25,persistent=False)
        self.register_buffer('bands',torch.stack(((f>0)&(f<=1/32),(f>1/32)&(f<=1/8),f>1/8)),persistent=False)
        self.coarse = phase_head(18)
        self.refine = phase_head(25)

    def phases(self, y, quality, power, *, use_action=True, use_noise=True):
        received = read_observation(y,quality,self.high_band,use_action=use_action,use_noise=use_noise)
        repeated, action, noise, disagreement, confidence, magnitude, angles, common = received
        fields = torch.stack((y.real,y.imag,magnitude.log1p(),quality.expand_as(y.real),
                              noise.log1p().expand_as(y.real),self.time.expand_as(y.real)),dim=2)
        fields = torch.cat((fields,angles.real,angles.imag,common.real,common.imag),dim=2)
        mixed = self.training and y.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            first = self.coarse(fields.flatten(0,1)).float().reshape_as(y.real)
        coarse = project(.40*first.tanh(),power,self.window,self.time)
        residual = (repeated-y*rotation(coarse))*confidence
        extra = torch.stack((coarse,residual.real,residual.imag,confidence,action.real-1.,
                             action.imag,disagreement.log1p()),dim=2)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            adjustment = self.refine(torch.cat((fields,extra),dim=2).flatten(0,1)).float().reshape_as(y.real)
        fine = project(.40*first.tanh()+.30*adjustment.tanh(),power,self.window,self.time)
        return coarse, fine, received[:5]


def generic_phase_observation(signal, rng):
    dtype, device = signal.real.dtype, signal.device
    batch, chirps, length = signal.shape
    phase = 2*torch.pi*torch.rand((batch,chirps,2,1),device=device,dtype=dtype,generator=rng)
    rate = .3+4.7*torch.rand((batch,1,2,1),device=device,dtype=dtype,generator=rng)
    amount = .24*torch.rand((batch,chirps,2,1),device=device,dtype=dtype,generator=rng)
    time = torch.arange(length,device=device,dtype=dtype)/length
    drift = (amount*(phase+2*torch.pi*rate*time).sin()).sum(-2)
    return varied_observation(signal*rotation(drift),rng)


def weak_phase_difference(ground, aligned, bands):
    difference = (aligned.roll(1,dims=0)*aligned.conj()).angle()
    low_spectrum = torch.fft.fft(difference,norm='ortho')*(bands[0]|bands[1])
    delta = (.25*torch.fft.ifft(low_spectrum,norm='ortho').real).clamp(-.16,.16)
    return ground*(rotation(delta)-1.)


def objective(model, tracker, iq, rng, *, with_identity, recipe, step):
    clean = complex_iq(iq)
    aligned, source_rotation, _, source_rms = coordinates(tracker,iq)
    delta = weak_phase_difference(clean,aligned,model.bands)
    wave = paired_inputs(clean,recipe.perturb,rng,delta,with_identity=with_identity)
    measured = wave.flatten(0,1)
    y, frame, quality, rms = coordinates(tracker,real_iq(measured))
    first, angle, observation = model.phases(y,quality,measured.abs().square())
    target = torch.stack((clean,clean,clean+delta),dim=1).flatten(0,1)/(frame*rms)
    coarse, fine = y*rotation(first), y*rotation(angle)
    error = torch.fft.fft(torch.stack((coarse-target,fine-target)),norm='ortho').abs().square()
    signal = torch.fft.fft(target,norm='ortho').abs().square()
    reconstruction = torch.stack([(error[..., band].mean(-1)/(.002+signal[...,band].mean(-1))).log1p().mean((1,2))
                                  for band in model.bands]).mean(0)
    repeated, _, noise, _, confidence = observation
    movement = (fine-y).abs().square()
    agreement = (fine-repeated).abs().square()-(y-repeated).abs().square()
    observation_loss = ((movement+.25*agreement)*confidence/(.025+noise)).mean()
    restored = (measured*rotation(angle)).unflatten(0,wave.shape[:2])
    basis = source_rotation*source_rms
    weak_loss = transfer_risk((restored[:,2]-restored[:,0])/basis,delta/basis)
    scale = min(1.,max(0.,(step-4000)/8000.))
    total = (.55-.20*scale)*reconstruction[0]+(.45+.20*scale)*reconstruction[1]
    total = total+(.035+.065*scale)*observation_loss+(.10+.25*scale)*weak_loss
    return total, torch.cat((reconstruction,observation_loss.view(1),weak_loss.view(1)))
