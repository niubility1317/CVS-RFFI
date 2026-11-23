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

def band_confidence(a, b, pa, pb, bands):
    """Held-group improvement of the half edit and a conservative best scalar."""
    fft = lambda x: torch.fft.fft(x, norm='ortho')
    gap = fft(b-a)
    energy = (gap.abs().square() @ bands.T).clamp_min(1e-8)
    gains, doses = [], []
    for proposed, before, sign in ((pa,a,1.),(pb,b,-1.)):
        change = fft(proposed-before)
        cross = (sign * (gap.conj()*change).real) @ bands.T
        square = change.abs().square() @ bands.T
        gains.append((cross-.25*square)/energy)
        doses.append((cross / square.clamp_min(1e-10)).clamp(0.,1.))
    return torch.minimum(*gains), torch.minimum(*doses)


def band_support(prior, captured, quality, bands, selectors):
    folds = []
    for start in (0,1):
        y, q = captured[:, start::2], quality[:, start::2]
        measured = torch.fft.fft(y, norm='ortho')
        moment, weights = spectral_contrast(measured,q)
        variance, _ = variance_interval(observed_noise_rms(y).square(),moment,weights,selectors)
        signal = prior.mean + prior.innovation(measured-prior.mean,variance@bands,True)
        proposal = y + zero_carrier(torch.fft.ifft(signal-measured,norm='ortho'))
        folds.append((y.mean(1),proposal.mean(1)))
    return band_confidence(folds[0][0],folds[1][0],folds[0][1],folds[1][1],bands)


def band_doses(old, gain, optimum, threshold):
    """Return hard-band, tapered weak-band, and support-limited boost doses."""
    base = .5 * old[:,None].to(gain.dtype)
    trust = (2*(gain-threshold)/(1-threshold).clamp_min(1e-5)).clamp(0.,1.)
    hard = torch.maximum(base,.5*(gain>threshold).to(gain.dtype))
    taper = torch.maximum(base,.5*trust)
    boost = taper + trust * (optimum-.5).clamp_min(0.) * old[:,None]
    return hard,taper,boost

def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    sys.path.insert(0,str(EXACT/'code'))
    from leo_practical.channel import ChannelStream
    import inspect
    assert 'return_operator' in inspect.signature(ChannelStream.process).parameters
    torch.set_num_threads(2)
    (training,cal,held),contract=setup(rt,'cpu')
    model,old_threshold=load_model(Path(rt['run_root']),RECIPES['rank'],contract,'cpu')
    bands=model.bands.clone()
    bands[:,0]=0.
    tracker=PacketRestorer('consensus').eval()
    def track(iq):
        return tuple(torch.cat(items) for items in zip(*(coordinates(tracker,v) for v in iq.split(40)),strict=True))
    def evidence(frames,prior=model.clean_detail):
        a,b=zip(*(band_support(prior,y,q,bands,model.selectors)
                    for y,q in zip(frames[0].split(64),frames[2].split(64),strict=True)),strict=True)
        return torch.cat(a),torch.cat(b)
    def describe(values,old,cut):
        gain,a=values
        armed=gain>cut
        dose=band_doses(old,gain,a,cut)
        return dict(n=len(gain),old=int(old.sum()),bands=armed.sum(0).tolist(),
                    added_band=(armed&~old[:,None]).sum(0).tolist(),
                    added_any=int((armed.any(1)&~old).sum()),
                    boosted_old=(dose[2][old]>.5).sum(0).tolist(),
                    mean_doses=[x.mean(0).tolist() for x in dose])
    cal_frame,held_frame=track(cal),track(held)
    cut=evidence(cal_frame)[0].amax(0)+1e-5
    held_counts=describe(evidence(held_frame),corruption_score(held_frame[0])>old_threshold,cut)
    fitting=track(torch.cat([rows[:100] for rows in training.chunk(len(SOURCE))]))
    leave_rx={}
    for i,site in enumerate(SOURCE):
        keep=torch.arange(500)//100!=i
        use=torch.arange(2500)//500!=i
        test=slice(i*100,(i+1)*100)
        prior=SharedSpectralPrior()
        prior.fit(torch.fft.fft(fitting[0][keep],norm='ortho'))
        limit=evidence(tuple(x[use] for x in cal_frame),prior)[0].amax(0)+1e-5
        frame=tuple(x[test] for x in held_frame)
        leave_rx[site]=describe(evidence(frame,prior),corruption_score(frame[0])>old_threshold,limit)
    source=Path(rt['source_root'])
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held_idx=partition(rx)
    take=np.concatenate([rows[:30] for rows in np.split(held_idx,len(SOURCE))])
    out={}
    with ProcessPoolExecutor(max_workers=6) as pool,torch.no_grad():
        for route in ('full','residual'):
            for scene in (VIEWS[1],VIEWS[3],VIEWS[-1]):
                inputs=((route,scene,raw[i],rx[i],ids[i]) for i in take)
                pairs=list(pool.map(replay,inputs,chunksize=6))
                iq,cleaned,_=(tensor(np.stack(items),'cpu') for items in zip(*pairs,strict=True))
                frame=track(iq)
                observed,rotation,q,rms=frame
                old=corruption_score(observed)>old_threshold
                gain,a=evidence(frame)
                state=describe((gain,a),old,cut)
                estimate=torch.cat([model(y,quality,phase)[-1] for y,quality,phase in
                                   zip(observed.split(16),q.split(16),rotation.split(16),strict=True)])
                edit=torch.fft.fft(estimate-observed,norm='ortho')
                target=torch.fft.fft(complex_iq(cleaned)/(rotation*rms)-observed,norm='ortho')
                base=(.5*old[:,None]).float().expand(len(old),4)
                half=torch.full_like(base,.5)
                arms=(base,half,*band_doses(old,gain,a,cut))
                result={}
                mask=bands.bool()
                for k,dose in zip(('original','open_half','band_half','band_taper','band_boost'),arms,strict=True):
                    pred=edit*(dose@bands)[:,None]
                    result[k]=[float((pred[...,m]-target[...,m]).abs().square().sum()/
                                           target[...,m].abs().square().sum().clamp_min(1e-12)) for m in mask]
                out.setdefault(route,{})[scene]=dict(support=state,normalized_noise_error=result)
    print(json.dumps(dict(packets=len(take),source_only=True,held_clean=held_counts,
                          thresholds=cut.tolist(),held_receiver_clean=leave_rx,actual_trace_used_only_after_freeze=True,
                          F_reads=0,V_reads=0,TX_reads=0,target_reads=0,by_route=out)),flush=True)

if __name__=='__main__':
    with torch.no_grad():
        main()
