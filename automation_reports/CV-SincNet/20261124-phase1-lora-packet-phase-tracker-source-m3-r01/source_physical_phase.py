
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.packet_phase import PacketPhaseTracker
from experiments.pamf_lora_main.packet_physics import aligned_view
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.tail_pilot_source import MIRROR_MODES, MIRROR_RUN

base=Path('/home/szu2070436088/2510044040/CV-SincNet')
parent=base/'runs'/MIRROR_RUN
release=base/'releases'/'20261124-phase1-lora-packet-phase-tracker-source-m3-r01-release'
rt=read(release/'runtime.json')
activate(rt)
torch.set_num_threads(2)
source=Path(rt['source_root'])
contract=source_contract(source)
physical=read_frozen(parent,'full',MIRROR_MODES)[0]
assert read_frozen(parent,'residual',MIRROR_MODES)[0]==physical
raw, rx, ids, _=source_role(source,contract,'V')
positions={p:i for i,p in enumerate(ids.tolist())}
chosen=np.array([positions[p] for p in physical]).reshape(len(SOURCE),-1)[:,3::6].flatten()
assert [sum(rx[chosen]==key) for key in SOURCE]==[30]*len(SOURCE)
rows=[(raw[i],rx[i],ids[i]) for i in chosen]
cache=np.load(base/'runs'/DATA_RUN/'source_views'/'V.npy',mmap_mode='r',allow_pickle=False)
tracker=PacketPhaseTracker().eval()
lims=torch.tensor([.5,.9,.99])

def methods(iq):
    plain=tracker.base.track(iq)
    quadratic=tracker.curvature(plain[0],plain[2])
    local=tracker.refine(plain)
    joint=tracker.refine(plain,quadratic)
    return dict(base=plain[1],local=local[1],joint=joint[1]),quadratic

def rate_rot(rot):
    return rot[...,128]*rot[...,127].conj()

def q(x):
    return x.flatten().quantile(lims).tolist()

@torch.no_grad()
def measure(y,n, include_tail):
    noisy, gamma=methods(y)
    quiet, ref_gamma=methods(y-n)
    clean_rate={name:rate_rot(rot) for name,rot in quiet.items()}
    reference=clean_rate['local']
    results={}
    for name, rot in noisy.items():
        current=rate_rot(rot)
        convert=256000/(2*torch.pi)
        results[name]=dict(noise_sensitivity_Hz=q((current * clean_rate[name].conj()).angle().abs()*convert),
                           relative_quiet_local_Hz=q((current * reference.conj()).angle().abs()*convert))
    details = dict(linear=results,quadratic_noise_change_radians=q((gamma-ref_gamma).abs()),
                absolute_noisy_quadratic_radians=q(gamma.abs()),
                absolute_quiet_quadratic_radians=q(ref_gamma.abs()))

    if include_tail:
        base_error = (rate_rot(noisy['base']) * clean_rate['base'].conj()).angle().abs() * convert
        local_error = (rate_rot(noisy['local']) * clean_rate['local'].conj()).angle().abs() * convert
        score = tracker.base.track(y)[2]
        selected = base_error.flatten().topk(max(1, round(base_error.numel()*.05))).indices
        before = base_error.flatten()[selected]
        after = local_error.flatten()[selected]
        tails = dict(chirps=len(selected), q_lower_than_02=int((score.flatten()[selected] < .2).sum()),
                     improves=int((after < before).sum()), base_Hz=q(before), local_Hz=q(after),
                     magnitude_q=q(score.flatten()[selected]))
        details['same_fixed_worst_5percent'] = tails
    return details

@torch.no_grad()
def recover_peak(iq):
    z = torch.complex(iq[...,0,:], iq[...,1,:])
    nominal = tracker.base.reference.chirp
    basis = z * nominal.conj()
    bins = torch.fft.fft(basis,n=2048).abs()
    peak, index = bins.max(-1)
    initial = tracker.base.track(iq)
    q = initial[2]
    anchor = q.argmax(-1,keepdim=True)
    center = index.gather(-1,anchor)
    width = 32  # 4 kHz at the fixed 125 Hz FFT-bin spacing.
    gap = (index-center+1024)%2048-1024
    anchor_count = ((gap.abs()<=width) & (q>=.45)).sum(-1,keepdim=True)
    shift = (torch.arange(2048)[None,None,:]-center[...,None]+1024)%2048-1024
    allowed = shift.abs()<=width
    candidate_peak, candidate_index = bins.masked_fill(~allowed,-1).max(-1)
    norm = z.abs().square().mean(-1).clamp_min(1e-12)
    candidate_q = (candidate_peak/256).square()/norm
    use = (q<.2) & (anchor_count>=2) & (gap.abs()>width) & (candidate_peak>=.35*peak) & (candidate_q>=.015)
    k = (candidate_index+1024)%2048-1024
    time = torch.arange(256)
    angle = 2*torch.pi*k[...,None]*time/2048
    rot = nominal * torch.polar(torch.ones_like(angle), angle)
    rz = z*rot.conj()
    dc = rz.mean(-1,keepdim=True)
    polarity = torch.where(dc.abs()>1e-10, dc/dc.abs().clamp_min(1e-10), torch.ones_like(dc))
    rz = rz*polarity.conj()
    rot = rot*polarity
    replacement = (torch.where(use[...,None], rz,initial[0]), torch.where(use[...,None],rot,initial[1]),
                   torch.where(use,candidate_q,q), initial[3],initial[4])
    return tracker.refine(replacement)[1], use

@torch.no_grad()
def probe(y,e):
    original = tracker.base.track(y)
    local = tracker.refine(original)[1]
    selected, use = recover_peak(y)
    quiet = rate_rot(tracker.refine(tracker.base.track(y-e))[1])
    convert = 256000/(2*torch.pi)
    before = (rate_rot(local)*quiet.conj()).angle().abs()*convert
    after = (rate_rot(selected)*quiet.conj()).angle().abs()*convert
    return dict(chirps=before.numel(), selected=int(use.sum()),
                over_400_Hz=[int((before>400).sum()),int((after>400).sum())],
                over_1000_Hz=[int((before>1000).sum()),int((after>1000).sum())],
                whole_Hz=[q(before),q(after)],
                selected_before_Hz=q(before[use]) if use.any() else [],
                selected_after_Hz=q(after[use]) if use.any() else [],
                selected_improved=int(((after<before)&use).sum()))

results={}
for route in ('full','residual'):
    results[route]={}
    with ProcessPoolExecutor(max_workers=4) as workers:
        for scene in ('practical_high','practical_low_urban'):
            values=list(workers.map(partial(replay,route,scene),rows,chunksize=8))
            wave, noise=(np.stack(x) for x in zip(*values))
            if route=='full':
                assert np.allclose(wave,cache[VIEWS.index(scene),chosen],atol=8e-7,rtol=2e-6)
            else:
                check=np.stack([aligned_view(w,scene,key,identifier,'source-V',processing_route=route)
                                for w,key,identifier in rows])
                assert np.allclose(wave,check,atol=8e-7,rtol=2e-6)
            y,e=tensor(wave,'cpu'),tensor(noise,'cpu')
            details=measure(y,e,scene=='practical_low_urban')
            if scene=='practical_low_urban':
                details['peak_probe']=probe(y,e)
            results[route][scene]=details
print(json.dumps(dict(packets=len(rows),quantiles=[.5,.9,.99],routes=results,
                      exact_added_noise_used_only_to_define_no_added_noise_reference=True,
                      no_labels=True, target_reads=0, decisions_frozen_in_parent=MIRROR_RUN)),flush=True)
