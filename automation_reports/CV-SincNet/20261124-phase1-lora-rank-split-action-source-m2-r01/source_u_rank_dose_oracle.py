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
from experiments.pamf_lora_main.spectral_covariance_source import RUN, RECIPES
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

def component(spectrum, name):
    mean=spectrum.mean(1,keepdim=True)
    if name=='common':
        return mean
    if name=='private':
        return spectrum-mean
    return spectrum

def risk(edit, truth, selector):
    edit, truth=edit[...,selector],truth[...,selector]
    dot=(edit.conj()*truth).real.sum((1,2)).double()
    norm=edit.abs().square().sum((1,2)).double()
    energy=truth.abs().square().sum((1,2)).double()
    optimal=(dot/norm.clamp_min(1e-14)).clamp(0.,1.)
    return dot,norm,energy,optimal

def combine(results, masks):
    rows={}
    for group,mask in masks.items():
        if not mask.any():
            rows[group]={'packets':0}
            continue
        dot,norm,energy,optimal=(x[mask] for x in results)
        denom=energy.sum().clamp_min(1e-14)
        rows[group]=dict(packets=int(mask.sum()),mean_dose=float(optimal.mean()),
                         nonpositive=int((dot<=0).sum()),
                         directional_alignment_cosine=float(dot.sum()/(norm.sum()*denom).sqrt().clamp_min(1e-14)),
                         half_recovered_noise_energy=float((dot-.25*norm).sum()/denom),
                         optimal_recovered_noise_energy=float((2*optimal*dot-optimal.square()*norm).sum()/denom))
    return rows

def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    sys.path.insert(0,str(EXACT/'code'))
    from leo_practical.channel import ChannelStream
    import inspect
    assert 'return_operator' in inspect.signature(ChannelStream.process).parameters
    torch.set_num_threads(2)
    source=Path(rt['source_root'])
    contract=source_contract(source)
    model,cut=load_model(Path(rt['run_root']),RECIPES['rank'],contract,'cpu')
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held=partition(rx)
    take=np.concatenate([block[:30] for block in np.split(held,len(SOURCE))])
    tracker=PacketRestorer('consensus').eval()
    frequency=(256*torch.fft.fftfreq(256)).round().abs()
    bands={'1_to_8':(frequency>0)&(frequency<=8),'9_to_32':(frequency>8)&(frequency<=32),
           '33_to_64':(frequency>32)&(frequency<=64),'above_64':frequency>64,'all_noncarrier':frequency>0}
    out={}
    with ProcessPoolExecutor(max_workers=5) as pool,torch.no_grad():
        for route in ('full','residual'):
            for scene in (VIEWS[1],VIEWS[3],VIEWS[-1]):
                items=((route,scene,raw[i],rx[i],ids[i]) for i in take)
                pairs=list(pool.map(replay,items,chunksize=6))
                iq,cleaned,_=(tensor(np.stack(values),'cpu') for values in zip(*pairs,strict=True))
                current,rotation,q,rms=coordinates(tracker,iq)
                predicted=torch.cat([model(y,r,p)[-1] for y,r,p in zip(current.split(16),q.split(16),
                                                                   rotation.split(16),strict=True)])
                desired=complex_iq(cleaned)/(rotation*rms)-current
                edit=torch.fft.fft(predicted-current,norm='ortho')
                truth=torch.fft.fft(desired,norm='ortho')
                active=corruption_score(current)>cut
                masks={'all':torch.ones_like(active),'bypass':~active,'old_active':active}
                specs={}
                for kind in ('common','private','total'):
                    e,v=component(edit,kind),component(truth,kind)
                    specs[kind]={name:combine(risk(e,v,k),masks) for name,k in bands.items()}
                out.setdefault(route,{})[scene]=specs
    print(json.dumps(dict(packets=len(take),source_role='held_U_s',target_reads=0,F_reads=0,V_reads=0,TX_reads=0,
        per_band_oracle_used_for_fit=False, exact_noise_only_diagnostic=True,by_route=out)),flush=True)

if __name__=='__main__':
    main()
