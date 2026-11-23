from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys

import numpy as np
import torch
from scipy.signal import resample_poly

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.spectral_covariance_source import RECIPES, RUN
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.data import FS_HZ, SOURCE, VIEWS, WINDOW_START
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.packet_physics import (CHIRPS, CONTEXT, RESIDUAL_LATENCY, channel,
    chirp_grid, chirp_samples, chirp_windows, hardware, pack_windows, residual_channel)
from experiments.pamf_lora_main.runner import activate

BASE = Path('/home/szu2070436088/2510044040/CV-SincNet')
CURRENT = BASE/'releases'/f'{RUN}-release'
EXACT = BASE/'releases'/'20261123-phase1-lora-dura-exact-action-source-m2-r01-release'
MODES = ('received', 'known_noise_removed', 'known_nonlinear_phase_removed', 'known_phase_and_noise_removed',
         'known_diagonal_inverse', 'known_diagonal_inverse_and_noise_removed')

def remove_known_receiver_terms(y, meta, trace):
    signal, noise = (trace[k] for k in ('pre_agc_signal', 'pre_agc_noise'))
    a, b = (trace[k] for k in ('operator_main', 'operator_image'))
    gain = 10**(meta['agc_gain_db']/20)
    if not np.array_equal((gain*(signal+noise)).astype(y.dtype), y):
        raise ValueError('Trace and gain must recover the registered waveform')
    if not np.all(np.abs(a) > np.abs(b)):
        raise ValueError('Invalid diagonal IQ transformation')
    return y, (gain*signal).astype(y.dtype), a, b

def paired_operations(y, signal, a, b):
    time = np.arange(a.shape[-1],dtype=np.float64)
    time -= time.mean()
    phase = np.unwrap(np.angle(a), axis=-1)
    slope = np.sum(phase*time,axis=-1,keepdims=True)/np.sum(time**2)
    nonlinear = phase-phase.mean(axis=-1,keepdims=True)-slope*time
    phase_only = np.exp(-1j*nonlinear)
    det = np.abs(a)**2-np.abs(b)**2
    def inverse(wave):
        return (a.conj()*wave-b*wave.conj())/det
    return y, signal, y*phase_only, signal*phase_only, inverse(y), inverse(signal)

def replay(item):
    route,scene,raw,rx,pid = item
    from leo_practical.channel import ChannelStream, normalize_input, sample_geometry, stable_seed
    from leo_practical.residual import ResidualChannel
    tx,_ = normalize_input(raw)
    namespace = 'source-U'
    receiver = hardware(scene, f'{namespace}/{rx}')
    parts = (namespace,scene,rx,str(pid))
    if route == 'full':
        cfg = channel(scene)
        stream = ChannelStream(cfg, stable_seed(2026112207,*parts), receiver)
        y,meta,trace = stream.process(tx[CONTEXT:],preceding_context=tx[:CONTEXT],
                                      return_traces=True,return_operator=True)
        latency = meta['numerical_filter_latency_samples']
        if meta['channel_equalization_applied'] or latency != cfg.fractional_delay_half_length:
            raise ValueError('Expected the aligned full channel without equalization')
        offset = latency-CONTEXT
        starts = np.asarray(CHIRPS)*1024+WINDOW_START-125+offset
        idx = starts[:,None]+np.arange(1250)
        windows = (a[idx] for a in remove_known_receiver_terms(y,meta,trace))
        waves = paired_operations(*windows)
        aligned = [resample_poly(x,up=32,down=125,axis=-1)[:,32:288] for x in waves]
        if not np.array_equal(aligned[0],chirp_samples(y,offset)):
            raise ValueError('Registered full crop is unchanged')
    else:
        cfg = residual_channel(scene)
        geometry = sample_geometry(cfg,np.random.default_rng(stable_seed(2026112207,*parts,'geometry')))
        start = 32+RESIDUAL_LATENCY
        grids = []
        for chirp,grid in zip(CHIRPS,chirp_grid(tx),strict=True):
            stream = ResidualChannel(cfg,stable_seed(2026112207,*parts,'chirp',chirp),receiver,geometry)
            y,meta,trace = stream.process(grid[start:start+256],preceding_context=grid[:start],
                                          return_traces=True,return_operator=True)
            if meta['channel_equalization_applied'] or meta['numerical_filter_latency_samples'] != RESIDUAL_LATENCY:
                raise ValueError('Expected the registered residual snapshot without equalization')
            grids.append(paired_operations(*remove_known_receiver_terms(y,meta,trace)))
        aligned = [np.stack(rows) for rows in zip(*grids,strict=True)]
    received,signal=aligned[:2]
    gain=np.sqrt(np.mean(np.abs(received)**2))
    # The noise-free reference stays in the noisy observation's absolute scale.
    noise_free=(np.stack((signal.real,signal.imag),axis=1)/gain).astype(np.float32)
    return pack_windows(received),noise_free,chirp_windows(raw)

BANDS = ('1_to_8','9_to_32','33_to_64')
FREQ=(256*torch.fft.fftfreq(256)).round().abs()
BINS=torch.nonzero((FREQ>0)&(FREQ<=64)).flatten()
N=BINS.numel()
SUPPORT=torch.cat([FREQ[BINS]]*2)
BAND_SELECTORS=dict(zip(BANDS,((SUPPORT<=8),(SUPPORT>8)&(SUPPORT<=32),SUPPORT>32),strict=True))
TRACKER=PacketRestorer('consensus').eval()

def vector(wave):
    spec=torch.fft.fft(wave,norm='ortho')
    near=spec.index_select(-1,BINS)
    return torch.cat((near.real,near.imag),dim=-1),spec

def report_row(store,route,scene,name,output,target,clean):
    result=store.setdefault(route,{}).setdefault(scene,{}).setdefault(name,{})
    for ref,t in (('same_observation_without_added_noise',target),('matched_clean_shape',clean)):
        for common in (True,False):
            def part(x):
                m=x.mean(1,keepdim=True)
                return m if common else x-m
            for b,select in BAND_SELECTORS.items():
                x,y=part(output)[...,select],part(t)[...,select]
                error=(x-y).square().sum().item()
                power=y.square().sum().item()
                product=(x*y).sum().item()
                slot=result.setdefault(ref,{}).setdefault('common' if common else 'private',{}).setdefault(b,[0.,0.,0.])
                slot[0]+=error
                slot[1]+=power
                slot[2]+=product

def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    sys.path.insert(0,str(EXACT/'code'))
    from leo_practical.channel import ChannelStream
    import inspect
    assert 'return_operator' in inspect.signature(ChannelStream.process).parameters
    torch.set_num_threads(2)
    root=Path(rt['source_root'])
    contract=source_contract(root)
    models, thresholds=zip(*(load_model(Path(rt['run_root']),recipe,contract,'cpu')
                             for recipe in RECIPES.values()),strict=True)
    assert len(set(thresholds))==1
    raw,rx,ids,_=source_role(root,contract,'U_s')
    _,_,held=partition(rx)
    sample=np.concatenate([rows[:30] for rows in np.split(held,len(SOURCE))])
    sums={}
    estimates={}
    with ProcessPoolExecutor(max_workers=4) as pool,torch.no_grad():
        for route in ('full','residual'):
            for scene in ('clean',VIEWS[3],VIEWS[4],VIEWS[-1]):
                if scene=='clean':
                    values=[chirp_windows(raw[i]) for i in sample]
                    obs=tensor(np.stack(values),'cpu')
                    signal,clean=obs,obs
                else:
                    items=((route,scene,raw[i],rx[i],ids[i]) for i in sample)
                    pairs=list(pool.map(replay,items,chunksize=6))
                    obs,signal,clean=(tensor(np.stack(rows),'cpu') for rows in zip(*pairs,strict=True))
                y,p,q,r=coordinates(TRACKER,obs)
                shape=vector(coordinates(TRACKER,clean)[0])[0].double()
                goal=vector(complex_iq(signal)/(p*r))[0].double()
                v=vector(y)[0].double()
                active=corruption_score(y)>thresholds[0]
                estimates.setdefault(route,{})[scene]=dict(active=int(active.sum()))
                report_row(sums,route,scene,'direct',v,goal,shape)
                guesses={k:y.clone() for k in ('diagonal_half','rank_half','rank_diagonal_half')}
                ix=torch.where(active)[0]
                for idx in ix.split(15):
                    if not len(idx):
                        continue
                    x=y[idx]
                    _,diagonal=models[0](x,q[idx],p[idx])
                    _,rank,_,_,same_diagonal=models[1](x,q[idx],p[idx],controls=True)
                    for name,value in zip(guesses,(diagonal,rank,same_diagonal),strict=True):
                        guesses[name][idx]=torch.lerp(x,value,.5)
                for name,value in guesses.items():
                    report_row(sums,route,scene,name,vector(value)[0].double(),goal,shape)
    for scenes in sums.values():
        for methods in scenes.values():
            for m in methods.values():
                for ref in m.values():
                    for part in ref.values():
                        for band,(error,power,inner) in tuple(part.items()):
                            part[band]=dict(relative_error=error/power,real_gain=inner/power)
    print(json.dumps(dict(packets=len(sample),source_role='held_U_s',estimates=estimates,
                          target_reads=0,F_reads=0,V_reads=0,TX_reads=0,
                          exact_noise_reference_consumed_by_model=False,
                          threshold=thresholds[0],by_route=sums)),flush=True)

if __name__=='__main__':
    main()
