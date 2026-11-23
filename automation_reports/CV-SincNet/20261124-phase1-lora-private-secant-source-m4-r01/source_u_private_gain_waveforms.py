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
from experiments.pamf_lora_main.private_secant_source import RUN, RECIPES
from experiments.pamf_lora_main.mirror_refine_source import load_model
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


def components(x):
    spectrum=torch.fft.fft(x,norm='ortho')
    shared=spectrum.mean(1,keepdim=True)
    return {'shared':shared,'private':spectrum-shared}


def summarize(prediction, target):
    by_packet=(1,2)
    ref=target.abs().square().sum(by_packet)
    cross=(prediction*target.conj()).sum(by_packet)
    valid=ref>1e-12
    energy=(prediction-target).abs().square().sum(by_packet)
    ratios=(cross[valid]/ref[valid])
    return dict(relative_error=float(energy.sum()/ref.sum().clamp_min(1e-12)),
                median_gain=float(ratios.real.median()),
                avg_gain=float(ratios.real.mean()),
                median_quadrature=float(ratios.imag.abs().median()),
                median_phase_rad=float(ratios.angle().abs().median()))


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
    loaded=[load_model(Path(rt['run_root']),recipe,contract,'cpu') for recipe in RECIPES.values()]
    (point,threshold),(candidate,matched)=loaded
    if threshold!=matched:
        raise ValueError('Different clean source U thresholds')
    tracker=PacketRestorer('consensus').eval()
    raw,rx,ids,_=source_role(Path(rt['source_root']),contract,'U_s')
    _,_,ix=partition(rx)
    take=np.concatenate([chunk[:30] for chunk in np.split(ix,len(SOURCE))])
    masks=point.bands.bool().clone()
    masks[:,0]=False
    bands=dict(zip(('1_to_8','9_to_32','33_to_64','65_to_128'),masks,strict=True))

    def coords(iq):
        chunks=[coordinates(tracker,p) for p in iq.split(24)]
        return tuple(torch.cat(x) for x in zip(*chunks,strict=True))

    def read_half(model, y, q, rotation, active):
        result=y.clone()
        for indices in torch.where(active)[0].split(16):
            if len(indices):
                value=model(y[indices],q[indices],rotation[indices])[-1]
                result[indices]=torch.lerp(y[indices],value,.5)
        return result

    def record_readout(output, references):
        parts=components(output)
        return {refname:{component:{key:summarize(parts[component][...,mask],ref[component][...,mask])
                                     for key,mask in bands.items()} for component in ref}
                for refname,ref in references.items()}

    tables={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            by_route={}
            for scene in (VIEWS[3],VIEWS[-1]):
                items=((route,scene,raw[i],rx[i],ids[i]) for i in take)
                received,uncorrupted,clean=(tensor(np.stack(p),'cpu') for p in
                                           zip(*pool.map(replay,items,chunksize=6),strict=True))
                y,rotation,q,rms=coords(received)
                source_clean=coords(clean)[0]
                noise_free=complex_iq(uncorrupted)/(rotation*rms)
                references={n:components(w) for n,w in (('matched_noise_free',noise_free),
                                                         ('matched_source_clean_shape',source_clean))}
                active=corruption_score(y)>threshold
                outputs={'direct':y}
                for model,arm in ((point,'paired_half'),(candidate,'private_gain_half')):
                    outputs[arm]=read_half(model,y,q,rotation,active)
                metrics={name:record_readout(wave,references) for name,wave in outputs.items()}
                by_route[scene]=dict(active=int(active.sum()),metrics=metrics)
            tables[route]=by_route
    print(json.dumps(dict(n=len(take),source_role='held_U_s',target_reads=0,F_reads=0,V_reads=0,TX_reads=0,
                          exact_simulator_use='diagnostic_reference_after_both_arms_frozen',by_route=tables)),flush=True)


if __name__=='__main__':
    main()
