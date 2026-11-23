from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys

import numpy as np
import torch
from scipy.signal import resample_poly

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import setup, partition
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.spectral_covariance_source import RUN, RECIPES
from experiments.pamf_lora_main.spectral_covariance import SharedSpectralPrior
from experiments.pamf_lora_main.variance_bound import spectral_contrast, variance_interval
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms
from experiments.pamf_lora_main.channel_blind import zero_carrier
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.data import FS_HZ, SOURCE, VIEWS, WINDOW_START
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.packet_physics import (CHIRPS, CONTEXT, RESIDUAL_LATENCY, channel,
    aligned_view, chirp_grid, chirp_samples, chirp_windows, hardware, pack_windows, residual_channel)
from experiments.pamf_lora_main.runner import activate

BASE = Path('/home/szu2070436088/2510044040/CV-SincNet')
CURRENT = BASE/'releases'/f'{RUN}-release'
EXACT = BASE/'releases'/'20261123-phase1-lora-dura-exact-action-source-m2-r01-release'
MODES = ('received', 'known_noise_removed', 'known_nonlinear_phase_removed', 'known_phase_and_noise_removed',
         'known_diagonal_inverse', 'known_diagonal_inverse_and_noise_removed')

"""Detect diffuse weak disturbances from disjoint receiver-only spectral evidence."""

import math

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit



class WeakDisturbanceFeatures(torch.nn.Module):
    """Jointly observe physical guard bands and dechirped repeat innovations."""

    def __init__(self):
        super().__init__()
        frequency = (256 * torch.fft.fftfreq(256)).round().abs()
        self.register_buffer('window', torch.hann_window(256))
        self.register_buffer('guards', torch.stack(((frequency >= 70) & (frequency <= 91),
                                                   (frequency >= 92) & (frequency <= 119))))
        self.register_buffer('innovations', torch.stack(((frequency >= 12) & (frequency <= 32),
                                                        (frequency >= 33) & (frequency <= 64),
                                                        (frequency >= 65) & (frequency <= 104))))

    @staticmethod
    def summaries(power, masks):
        """A robust floor and flatness for each band, then two packet order statistics."""
        summaries = []
        for mask in masks:
            band = power[..., mask]
            floor = band.median(-1).values.clamp_min(1e-12)
            total = band.mean(-1).clamp_min(1e-12)
            for scalar in (floor.log(), (floor/total).log()):
                ordered = scalar.sort(dim=1).values
                summaries.extend((ordered[:, 2], ordered[:, 4]))
        return summaries

    def forward(self, iq, observed, quality):
        raw = complex_iq(iq)
        raw_rms = raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        raw = (raw / raw_rms) * self.window / self.window.square().mean().sqrt()
        guard_power = torch.fft.fft(raw, norm='ortho').abs().square()
        spec = torch.fft.fft(observed, norm='ortho')
        weight = quality.square().clamp_min(1e-6)
        repeat_power = (spec - (spec * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True)).abs().square()
        # The receiver observes gradual floors and dispersion, not transmitter identity or receiver ID.
        a = self.summaries(guard_power, self.guards)
        b = self.summaries(repeat_power, self.innovations)
        return torch.stack(a+b, dim=1)


def monotone_scores(positive_inputs, negative_inputs):
    """Use paired generic corruptions for clean-versus-impaired logistic fitting."""
    midpoint = negative_inputs.median(dim=0).values
    p25,p75 = torch.quantile(negative_inputs, negative_inputs.new_tensor([.25,.75]),dim=0)
    dispersion = (p75-p25).clamp_min(.10)
    a = ((negative_inputs-midpoint)/dispersion).clamp(-10,10).double().numpy()
    b = ((positive_inputs-midpoint)/dispersion).clamp(-10,10).double().numpy()
    values = np.concatenate((a,b))
    target = np.concatenate((np.zeros(len(a)),np.ones(len(b))))
    weights = np.concatenate((np.full(len(a),.5/len(a)),np.full(len(b),.5/len(b))))
    def loss(state):
        w = state[:-1]
        z = values@w + state[-1]
        objective = np.sum(weights*(np.logaddexp(0,z)-target*z)) + .025 * np.dot(w,w)
        delta = weights*(expit(z)-target)
        gradient = np.concatenate((values.T@delta+.05*w,[delta.sum()]))
        return objective,gradient
    result=minimize(loss,np.zeros(values.shape[1]+1),method='L-BFGS-B',jac=True,
                    bounds=[(0,None)]*values.shape[1]+[(None,None)],options=dict(maxiter=150))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Generic weak-disturbance fit failed: {result.message}')
    return midpoint,dispersion,negative_inputs.new_tensor(result.x[:-1]),float(result.x[-1])


def apply_scores(row, midpoint, dispersion, weight, bias):
    return ((row-midpoint)/dispersion).clamp(-10,10) @ weight + bias


def generic_noise(iq, rng, amplitude=None):
    """Generic independent complex noise, either flat or smoothly colored."""
    size=iq.shape
    opts=dict(device=iq.device,generator=rng)
    draw=torch.complex(torch.randn(size,**opts),torch.randn(size,**opts))
    frequency=torch.fft.fftfreq(size[-1],device=iq.device)
    angle=2*torch.pi*(frequency-torch.rand((len(iq),1,1),**opts))
    depth=.95*torch.rand((len(iq),1,1),**opts)
    flat=torch.rand((len(iq),1,1),**opts)<.40
    profile=(1+torch.where(flat,0.,depth)*angle.cos()).sqrt()
    draw=torch.fft.ifft(torch.fft.fft(draw) * profile)
    if amplitude is None:
        amplitude=math.exp(math.log(.006))*torch.exp(math.log(.16/.006)*torch.rand((len(iq),1,1),**opts))
    return iq + amplitude * draw/2**.5

def extract(tracker, to_features, iq):
    x,_,q,_=coordinates(tracker,iq)
    return to_features(iq,x,q),corruption_score(x)

def joined(tracker, vector, iq):
    chunks=[extract(tracker,vector,part) for part in iq.split(80)]
    return tuple(torch.cat(list(v)) for v in zip(*chunks,strict=True))

def replay(row):
    raw,rx,pid,route,scene=row
    return aligned_view(raw,scene,rx,pid,'source-U',processing_route=route)

import torch


def support_gain(before_a, before_b, after_a, after_b, bands):
    """Conservative held-chirp validation of a candidate common waveform update."""
    fft=lambda x:torch.fft.fft(x,norm='ortho')
    difference=fft(before_b-before_a)
    emissions=(fft(after_a-before_a),fft(after_b-before_b))
    error=difference.abs().square()@bands.T
    gains=[]
    for delta,sign in zip(emissions,(1.,-1.),strict=True):
        cross=(2*sign*(difference.conj()*delta).real)@bands.T
        penalty=delta.abs().square()@bands.T
        gains.append((cross-penalty)/error.clamp_min(1e-8))
    return torch.minimum(*gains),.5*(gains[0]+gains[1])


def observed_support(prior, observed, coherence, bands, selectors):
    """Compute proof scores from independent odd/even chirp proposal fits."""
    half=[]
    for start in (0,1):
        y,q=observed[:,start::2],coherence[:,start::2]
        measured=torch.fft.fft(y,norm='ortho')
        contrast,w=spectral_contrast(measured,q)
        variance,_=variance_interval(observed_noise_rms(y).square(),contrast,w,selectors)
        noise=variance@bands
        prediction=prior.mean+prior.innovation(measured-prior.mean,noise,True)
        proposal=y+zero_carrier(torch.fft.ifft(prediction-measured,norm='ortho'))
        half.append((y.mean(1),torch.lerp(y,proposal,.5).mean(1)))
    confirm,gain=support_gain(half[0][0],half[1][0],half[0][1],half[1][1],bands)
    return torch.cat((confirm,gain),dim=1)

def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    torch.set_num_threads(3)
    (training,cal,held),contract=setup(rt,'cpu')
    tracker=PacketRestorer('consensus').eval()
    model,cut=load_model(Path(rt['run_root']),RECIPES['rank'],contract,'cpu')
    def track(x):
        return tuple(torch.cat(parts) for parts in zip(*(coordinates(tracker,q) for q in x.split(72)),strict=True))
    def scores(prior,frames):
        result=[]
        for y,q in zip(frames[0].split(72),frames[2].split(72),strict=True):
            d=observed_support(prior,y,q,model.bands,model.selectors)
            a,b=d.chunk(2,dim=1)
            result.append(torch.stack((a[:,0],a[:,:2].mean(1),a[:,1:3].mean(1),
                                         a[:,:3].mean(1),b[:,:2].mean(1),a.mean(1)),dim=1))
        return torch.cat(result)
    train=track(torch.cat([p[:100] for p in training.chunk(len(SOURCE))]))
    clean,cali=track(held),track(cal)
    rng=torch.Generator().manual_seed(2026113229)
    synthetic={d:track(real_iq(generic_noise(complex_iq(held),rng,amplitude=d))) for d in (.015,.03,.06,.12)}
    names=('both_1_8','both_1_32','both_9_64','both_1_64','mean_1_32','both_full')
    base=scores(model.clean_detail,cali)
    def stats(values,threshold,frames):
        active=values>threshold
        old=corruption_score(frames[0])>cut
        return dict(packets=len(values),old=int(old.sum()),ranges=values.quantile(torch.tensor([.1,.5,.9])).T.tolist(),
                    selected=active.sum(0).tolist(),extra=(active&~old[:,None]).sum(0).tolist())
    quantile=base.quantile(.999,dim=0)
    strict=base.amax(dim=0)+1e-5
    same={str(k):{name:stats(scores(model.clean_detail,f),lim,f)
                  for name,lim in (('maximum',strict),('p999',quantile),('positive',torch.zeros_like(strict)))}
                  for k,f in (('clean',clean),*synthetic.items())}
    leaves={}
    for i,site in enumerate(SOURCE):
        support=torch.ones(len(train[0]),dtype=torch.bool)
        support[i*100:(i+1)*100]=False
        c=torch.ones(len(cali[0]),dtype=torch.bool)
        c[i*500:(i+1)*500]=False
        single=slice(i*100,(i+1)*100)
        prior=SharedSpectralPrior()
        prior.fit(torch.fft.fft(train[0][support],norm='ortho'))
        bound=scores(prior,tuple(x[c] for x in cali)).amax(0)+1e-5
        leaves[site]={}
        for k,f in (('clean',clean),*synthetic.items()):
            held_frames=tuple(x[single] for x in f)
            leaves[site][str(k)]=stats(scores(prior,held_frames),bound,held_frames)
    raw,rx,ids,_=source_role(Path(rt['source_root']),contract,'U_s')
    _,_,held_ix=partition(rx)
    take=np.concatenate([q[:30] for q in np.split(held_ix,len(SOURCE))])
    real={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            for scene in (VIEWS[1],VIEWS[3],VIEWS[-1]):
                seq=((raw[i],rx[i],ids[i],route,scene) for i in take)
                y=tensor(np.stack(list(pool.map(replay,seq,chunksize=8))),'cpu')
                frames=track(y)
                x=scores(model.clean_detail,frames)
                real.setdefault(route,{})[scene]={name:stats(x,lim,frames) for name,lim in
                   (('maximum',strict),('p999',quantile),('positive',torch.zeros_like(strict)))}
    print(json.dumps(dict(names=names,source_only=True,held=len(held),calibration=len(cal),
                     fit_per_leaving_out_receiver=400,V_reads=0,F_reads=0,TX_reads=0,target_reads=0,
                     exact_truth_reads=0,source_U_real_after_freeze=True,strict=strict.tolist(),
                     same_rx=same,leave_rx=leaves,by_route=real)),flush=True)

if __name__=='__main__':
    main()
