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
from experiments.pamf_lora_main.coupled_echo_source import RUN
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

def prepare(clean):
    mean=clean.mean(0)
    value=clean-mean
    covariance=value.T@value/(len(value)-1)
    target=BAND_SELECTORS['1_to_8']
    maps={}
    for key in BANDS[1:]:
        supported=BAND_SELECTORS[key]
        cov=covariance[supported][:,supported].contiguous()
        cross=covariance[supported][:,target].contiguous()
        eig,axes=torch.linalg.eigh(.7*cov+.3*torch.diag(cov.diagonal()))
        maps[key]=(supported,eig.clamp_min(1e-8),axes,axes.T@cross)
    return mean,maps

def predict(mean,model,read,noise):
    guesses={}
    for key,(bins,eig,axes,cross) in model.items():
        var=noise[:,None]/6.
        guesses[key]=((((read-mean)[:,bins]@axes)/(var+eig))@cross)
    return guesses

def stat(delta, estimate, directly_observed):
    target_power=delta.square().sum().clamp_min(1e-10)
    base=(directly_observed-delta).square().sum()
    proposed=(estimate-delta).square().sum()
    return [float(target_power),float(base),float(proposed)]

def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    sys.path.insert(0,str(EXACT/'code'))
    from leo_practical.channel import ChannelStream
    import inspect
    assert 'return_operator' in inspect.signature(ChannelStream.process).parameters
    torch.set_num_threads(2)
    root=Path(rt['source_root'])
    raw,rx,ids,_=source_role(root,source_contract(root),'U_s')
    _,_,held=partition(rx)
    clean=tensor(np.stack([chirp_windows(raw[i]) for i in held]),'cpu')
    canonical=vector(coordinates(TRACKER,clean)[0])[0].mean(1).double()
    sample=np.concatenate([rows[:30] for rows in np.split(held,len(SOURCE))])
    folds=[prepare(canonical[np.flatnonzero(rx[held]!=r).tolist()]) for r in SOURCE]
    out={}
    target=BAND_SELECTORS['1_to_8']
    with ProcessPoolExecutor(max_workers=5) as pool,torch.no_grad():
        for route in ('full','residual'):
            for scene in (VIEWS[3],VIEWS[4],VIEWS[-1]):
                pairs=list(pool.map(replay,((route,scene,raw[i],rx[i],ids[i]) for i in sample),chunksize=6))
                obs,signal,_=(tensor(np.stack(rows),'cpu') for rows in zip(*pairs,strict=True))
                y,p,q,r=coordinates(TRACKER,obs)
                x,spectrum=vector(y)
                x=x.mean(1).double()
                truth=vector(complex_iq(signal)/(p*r))[0].mean(1).double()
                sigma=spectrum[...,FREQ>80].abs().square().flatten(1).median(-1).values.double()/(2*np.log(2))
                batches=[]
                for i,(mean,model) in enumerate(folds):
                    select=slice(i*30,(i+1)*30)
                    terms=predict(mean,model,x[select],sigma[select])
                    a,b=tuple(terms.values())
                    concordance=(a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-10)
                    measured=(x[select]-mean)[:,target]
                    expected=(truth[select]-mean)[:,target]
                    batches.append((a,b,concordance,measured,expected))
                a,b,cosine,m,t=(torch.cat(rows) for rows in zip(*batches,strict=True))
                stats={}
                for name,allowed in (('all',torch.ones_like(cosine,dtype=torch.bool)),
                                      ('agree',cosine>=.25),('disagree',cosine<.25)):
                    vals={}
                    for k,est in (('near_anchor',a),('far_anchor',b),('equal_mixture',.5*(a+b))):
                        norm,raw_error,est_error=stat(t[allowed],est[allowed],m[allowed])
                        vals[k]=dict(raw_relative_error=raw_error/norm,estimate_relative_error=est_error/norm)
                    stats[name]=dict(n=int(allowed.sum()),data=vals)
                out.setdefault(route,{})[scene]=dict(stages=stats,cosine_median=float(cosine.median()))
    print(json.dumps(dict(packets=len(sample),source_role='held_U_s',held_receivers=list(SOURCE),targets_read=0,
                          F_reads=0,TX_reads=0,V_reads=0,provided_truth_to_estimate=False,
                          reference='received frame with exact added noise removed offline',by_route=out)),flush=True)

if __name__=='__main__':
    main()
