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
from experiments.pamf_lora_main.spectral_covariance_source import RUN
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


def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    torch.set_num_threads(3)
    (training,cal,held),contract=setup(rt,'cpu')
    tracker=PacketRestorer('consensus').eval()
    feature=WeakDisturbanceFeatures()
    rng=torch.Generator().manual_seed(2026113107)
    train,ts=joined(tracker,feature,training)
    calf,ca=joined(tracker,feature,cal)
    hf,hs=joined(tracker,feature,held)
    positive=torch.stack([joined(tracker,feature,real_iq(generic_noise(complex_iq(training),rng)))[0] for _ in range(2)])
    rng=torch.Generator().manual_seed(2026113137)
    low,ls=joined(tracker,feature,real_iq(generic_noise(complex_iq(held),rng,amplitude=.0075)))
    mid,ms=joined(tracker,feature,real_iq(generic_noise(complex_iq(held),rng,amplitude=.015)))
    out={}
    selectors={'guard_floor':torch.tensor([0,1,4,5]),'guard_flatness':torch.tensor([2,3,6,7]),
      'guard_all':torch.arange(8),'repeat_floor':torch.tensor([8,9,12,13,16,17]),
      'repeat_flatness':torch.tensor([10,11,14,15,18,19]),'repeat_all':torch.arange(8,20),'joint':torch.arange(20)}
    both=('guard_and_repeat','guard_flatness_and_repeat')
    fixed={}
    for k,where in selectors.items():
        fit=monotone_scores(positive[:,:,where].flatten(0,1),train[:,where])
        fixed[k]=(fit,float(apply_scores(calf[:,where],*fit).max())+.01)
    for i,site in enumerate(SOURCE):
        tr=torch.arange(len(train))//(len(train)//len(SOURCE))
        cr=torch.arange(len(calf))//(len(calf)//len(SOURCE))
        hr=torch.arange(len(hf))//(len(hf)//len(SOURCE))
        fits={}
        for k,where in selectors.items():
            fit=monotone_scores(positive[:,tr!=i][:,:,where].flatten(0,1),train[tr!=i][:,where])
            fits[k]=(fit,float(apply_scores(calf[cr!=i][:,where],*fit).max())+.01)
        samples={}
        for name,features in {'clean':hf,'amplitude_0.0075':low,'amplitude_0.015':mid}.items():
            alarms={}
            for k,where in selectors.items():
                fit,threshold=fits[k]
                alarms[k]=apply_scores(features[hr==i][:,where],*fit)>threshold
            alarms[both[0]]=alarms['guard_all']&alarms['repeat_all']
            alarms[both[1]]=alarms['guard_flatness']&alarms['repeat_all']
            samples[name]={k:int(v.sum()) for k,v in alarms.items()}
        out[site]=samples
    practical={}
    raw,rx,ids,_=source_role(Path(rt['source_root']),contract,'U_s')
    _,_,held_idx=partition(rx)
    take=np.concatenate([chunk[:45] for chunk in np.split(held_idx,len(SOURCE))])
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            for scene in (VIEWS[1],VIEWS[3],VIEWS[4],VIEWS[-1]):
                items=((raw[i],rx[i],ids[i],route,scene) for i in take)
                received=tensor(np.stack(list(pool.map(replay,items,chunksize=8))),'cpu')
                feature_rows,outer=joined(tracker,feature,received)
                alarms={}
                for k,where in selectors.items():
                    fit,threshold=fixed[k]
                    alarms[k]=apply_scores(feature_rows[:,where],*fit)>threshold
                alarms[both[0]]=alarms['guard_all']&alarms['repeat_all']
                alarms[both[1]]=alarms['guard_flatness']&alarms['repeat_all']
                old=outer>1.05*ca.max()
                practical.setdefault(route,{})[scene]={'old':int(old.sum()),
                    'gates':{k:dict(active=int(v.sum()),additional=int((v&~old).sum())) for k,v in alarms.items()}}
    print(json.dumps(dict(source_only=True,n_fit_per_fold=4000,n_cal_per_fold=2000,
                          held_per_fold=100,panel_per_view=225,fit_seed=2026113107,held_seed=2026113137,
                          target_reads=0,F_reads=0,V_reads=0,TX_reads=0,counts=out,by_route=practical)),flush=True)

if __name__=='__main__':
    main()
