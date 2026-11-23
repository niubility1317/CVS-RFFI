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
from experiments.pamf_lora_main.private_dose_source import RUN, MODES as DOSE_MODES, parent_models
from experiments.pamf_lora_main.private_dose import action_parts, component_swaps
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



def band_summary(value, reference, rms, selector):
    weight=rms.square()
    signal=reference[...,selector]
    predicted=value[...,selector]
    power=(signal.abs().square()*weight).sum((1,2))
    cross=(predicted*signal.conj()*weight).sum((1,2))
    valid=power>1e-12
    amp=cross[valid]/power[valid]
    error=((predicted-signal).abs().square()*weight).sum((1,2))
    return dict(relative_error=float(error.sum()/power.sum().clamp_min(1e-12)),
                median_gain=float(amp.real.median()),
                median_phase_rad=float(amp.angle().abs().median()),
                median_quadrature=float(amp.imag.abs().median()))


@torch.no_grad()
def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    sys.path.insert(0,str(EXACT/'code'))
    from leo_practical.channel import ChannelStream
    import inspect
    assert 'return_operator' in inspect.signature(ChannelStream.process).parameters
    torch.set_num_threads(2)
    (_,_,_),contract=setup(rt,'cpu',smoke=True)
    models,threshold=parent_models(Path(rt['run_root']),contract,'cpu')
    tracker=PacketRestorer('consensus').eval()
    raw,rx,ids,_=source_role(Path(rt['source_root']),contract,'U_s')
    _,_,ix=partition(rx)
    take=np.concatenate([chunk[:30] for chunk in np.split(ix,len(SOURCE))])
    bands=models[0].bands.clone()
    bands[:,0]=0
    masks=dict(zip(('1_to_8','9_to_32','33_to_64','65_to_128'),bands.bool(),strict=True))

    def coords(iq):
        chunks=[coordinates(tracker,p) for p in iq.split(24)]
        return tuple(torch.cat(x) for x in zip(*chunks,strict=True))

    tables={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            by_route={}
            for scene in (VIEWS[3],VIEWS[-1]):
                items=((route,scene,raw[i],rx[i],ids[i]) for i in take)
                received,uncorrupted,_=(tensor(np.stack(p),'cpu') for p in
                                         zip(*pool.map(replay,items,chunksize=6),strict=True))
                y,rotation,q,rms=coords(received)
                active=corruption_score(y)>threshold
                original=torch.fft.fft(y,norm='ortho')
                target=torch.fft.fft(complex_iq(uncorrupted)/(rotation*rms),norm='ortho')
                reference=action_parts(target,rms)
                deltas=[]
                for model in models:
                    wave=torch.zeros_like(y)
                    for selection in torch.where(active)[0].split(16):
                        if len(selection):
                            proposal=model(y[selection],q[selection],rotation[selection])[-1]
                            wave[selection]=torch.lerp(y[selection],proposal,.5)-y[selection]
                    deltas.append(torch.fft.fft(wave,norm='ortho'))
                variants=(torch.zeros_like(original),*deltas,*component_swaps(*deltas,rms,bands))
                rows={}
                for name,delta in zip((DOSE_MODES[0],*DOSE_MODES[2:]),variants,strict=True):
                    output=action_parts(original+delta,rms)
                    rows[name]={part:{key:band_summary(value,ref,rms,mask) for key,mask in masks.items()}
                                for part,value,ref in zip(('shared','private'),output,reference,strict=True)}
                by_route[scene]=dict(active=int(active.sum()),metrics=rows)
            tables[route]=by_route
    print(json.dumps(dict(n=len(take),source_role='held_U_s',target_reads=0,F_reads=0,V_reads=0,TX_reads=0,
                          exact_simulator_use='diagnostic_reference_only',by_route=tables)),flush=True)


if __name__=='__main__':
    main()
