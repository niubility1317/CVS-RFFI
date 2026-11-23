from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys

import numpy as np
import torch
from scipy.signal import resample_poly

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition, setup
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.spectral_covariance_source import RUN, RECIPES
from experiments.pamf_lora_main.mirror_refine_source import load_model
from experiments.pamf_lora_main.channel_blind import zero_carrier
from experiments.pamf_lora_main.variance_bound import spectral_contrast,variance_interval
from experiments.pamf_lora_main.weak_signal_probe import observed_noise_rms
from experiments.pamf_lora_main.spectral_covariance import SharedSpectralPrior
from experiments.pamf_lora_main.data import FS_HZ, SOURCE, VIEWS, WINDOW_START
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
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


def measure_alias(y, signal, noise_in_raw, tracker):
    y,phase,q,rms=coordinates(tracker,y)
    noise=noise_in_raw/(phase*rms)
    clean=complex_iq(signal)/(phase*rms)
    def ac(x):
        return torch.fft.fft(x-x.mean(1,keepdim=True),norm='ortho').abs().square()
    noise_power,signal_power,received_power=map(ac,(noise,clean,y))
    raw_frequency=(256*torch.fft.fftfreq(256)).round().abs()
    data={}
    for first,last in ((1,8),(9,32),(33,63)):
        low=torch.where((raw_frequency>=first)&(raw_frequency<=last))[0]
        high=(low+128)%256
        n_low,n_high=noise_power[...,low],noise_power[...,high]
        s_low,s_high=signal_power[...,low],signal_power[...,high]
        r_low,r_high=received_power[...,low],received_power[...,high]
        axes=(1,2)
        theoretical=(s_low-s_high).sum(axes).double()
        surrogate=(r_low-r_high).sum(axes).double()
        corr=torch.corrcoef(torch.stack((theoretical,surrogate)))[0,1]
        variance=theoretical.square().sum().clamp_min(1e-10)
        response=(n_low-n_high).sum()/n_low.sum().clamp_min(1e-10)
        data[f'{first}_to_{last}']=dict(source_alias_to_signal_ratio=float(s_high.sum()/s_low.sum()),
               unmatched_noise_in_relative_alias_contrast=float(response),
               packet_contrast_pearson=float(corr),
               relative_contrast_error=float((theoretical-surrogate).square().sum()/variance))
    return data

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
    raw,rx,ids,_=source_role(root,contract,'U_s')
    _,_,held=partition(rx)
    indices=np.concatenate([r[:30] for r in np.split(held,len(SOURCE))])
    tracker=PacketRestorer('consensus').eval()
    f=(256*torch.fft.fftfreq(256)).round().abs()
    response=.5+.5*torch.cos(torch.pi*((f-60)/8).clamp(0,1))
    out={}
    with ProcessPoolExecutor(max_workers=6) as pool,torch.no_grad():
        for route in ('full','residual'):
            for scene in (VIEWS[3],VIEWS[-1]):
                rows=((route,scene,raw[i],rx[i],ids[i]) for i in indices)
                iq,signal,clean=(tensor(np.stack(s),'cpu') for s in zip(*pool.map(replay,rows,chunksize=6),strict=True))
                noise=complex_iq(iq-signal)
                limited=torch.fft.ifft(torch.fft.fft(noise,norm='ortho')*response,norm='ortho')
                y=signal+real_iq(limited)
                out.setdefault(route,{})[scene]={
                   'registered':measure_alias(iq,signal,noise,tracker),
                   'occupied_noise_only':measure_alias(y,signal,limited,tracker)}
    print(json.dumps(dict(n=len(indices),route_do_not_use_to_choose_action=True,source_role='held_U_s',
                          F_reads=0,V_reads=0,TX_reads=0,target_reads=0,noise_truth_diagnostic_only=True,
                          by_route=out)),flush=True)

if __name__=='__main__':
    main()
