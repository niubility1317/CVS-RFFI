from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys

import numpy as np
import torch
from scipy.signal import resample_poly

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind_source import partition
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
    return np.stack([pack_windows(w) for w in aligned]), chirp_windows(raw)

def tracked(tracker, iq):
    wave, rotation, q, *_ = tracker.track(iq)
    return wave, rotation, q

def canonical(wave, tone):
    # Remove only the estimated constant complex tone and make sidebands unit-tone.
    return torch.fft.fft(wave*tone.conj()/tone.abs().square().clamp_min(.01),norm='ortho')

def components(w):
    c=w.mean(1,keepdim=True)
    return {'common':c, 'private':w-c}

def summarize(values,clean):
    tracker = PacketRestorer('consensus').eval()
    w = tensor(values,'cpu')
    received_w, received_rot, received_q = tracked(tracker,w[:,0])
    noiseless_w, noiseless_rot, noiseless_q = tracked(tracker,w[:,1])
    ground, _, clean_q = tracked(tracker,tensor(clean,'cpu'))
    original_noisy_in_cleaned_frame = complex_iq(w[:,0])*noiseless_rot.conj()
    base = components(canonical(ground,ground.mean(-1,keepdim=True)))
    bin = (256*torch.fft.fftfreq(256)).round().abs()
    bands = {'1_to_8':(bin>0)&(bin<=8),'9_to_32':(bin>8)&(bin<=32),
             '33_to_64':(bin>32)&(bin<=64),'above_64':bin>64}
    support = ((received_q>=.12)&(noiseless_q>=.12)&(clean_q>=.12)).all(1)
    if int(support.sum())<=80:
        raise ValueError('The matched comparison lacks carrier support')
    tracking_error = ((received_rot[...,128]*received_rot[...,127].conj()*
                       noiseless_rot[...,128].conj()*noiseless_rot[...,127]).angle() * FS_HZ/(2*np.pi)).abs()
    rows = dict(received=received_w, oracle_frame_noisy_tone=original_noisy_in_cleaned_frame,
                oracle_frame_true_tone=original_noisy_in_cleaned_frame, known_noise_removed=noiseless_w)
    rows.update({mode:tracked(tracker,w[:,i])[0] for i,mode in enumerate(MODES) if i>1})
    out={}
    for name, v in rows.items():
        # The noise-matched row uses phase and complex scale from the noise-free
        # frame. Estimating its phase from the noisy carrier would undo the control.
        tone=(noiseless_w if name=='oracle_frame_true_tone' else v).mean(-1,keepdim=True)
        z=components(canonical(v,tone))
        result={}
        for part,x in z.items():
            for band,mask in bands.items():
                a,b=x[support][...,mask],base[part][support][...,mask]
                power=b.abs().square().sum((1,2))
                inner=(a*b.conj()).sum((1,2))
                result.setdefault(part,{})[band]=dict(
                    relative_error=float((a-b).abs().square().sum()/power.sum()),
                    median_real_gain=float((inner.real/power.clamp_min(1e-10)).median()),
                    median_phase_rad=float(torch.atan2(inner.imag,inner.real).abs().median()))
        out[name]=result
    return dict(supported_packets=int(support.sum()),tracking_error_hz_p50_p90_p99=
                tracking_error.flatten().quantile(torch.tensor([.5,.9,.99])).tolist(),modes=out)

def main():
    rt=read(CURRENT/'runtime.json')
    activate(rt)
    sys.path.insert(0,str(EXACT/'code'))
    from leo_practical.channel import ChannelStream
    from leo_practical.residual import ResidualChannel
    import inspect
    for f in (ChannelStream.process,ResidualChannel.process):
        if 'return_operator' not in inspect.signature(f).parameters or EXACT not in Path(inspect.getfile(f)).parents:
            raise ValueError('Must use the frozen exact-action trace extension')
    torch.set_num_threads(2)
    source=Path(rt['source_root'])
    raw,rx,ids,_=source_role(source,source_contract(source),'U_s')
    _,_,held=partition(rx)
    sample=np.concatenate([rows[:30] for rows in np.split(held,len(SOURCE))])
    out={}
    with ProcessPoolExecutor(max_workers=5) as pool, torch.no_grad():
        for route in ('full','residual'):
            for scene in (VIEWS[3],VIEWS[4],VIEWS[-1]):
                items=((route,scene,raw[i],rx[i],ids[i]) for i in sample)
                pairs=list(pool.map(replay,items,chunksize=6))
                values,clean=(np.stack(rows) for rows in zip(*pairs,strict=True))
                out.setdefault(route,{})[scene]=summarize(values,clean)
    print(json.dumps(dict(packets=len(sample),source_role='held_U_s',reference='independently tracked clean, per-chirp unit tone',
          trace_parent=EXACT.name,target_reads=0,F_reads=0,V_reads=0,TX_reads=0,
          oracle_corrections='diagnostics only',by_route=out)),flush=True)

if __name__=='__main__':
    main()
