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
    (fit,cal,_),contract=setup(rt,'cpu')
    rank,old_threshold=load_model(Path(rt['run_root']),RECIPES['rank'],contract,'cpu')
    bands=rank.bands.clone()
    bands[:,0]=0.
    tracker=PacketRestorer('consensus').eval()
    def track(value):
        return tuple(torch.cat(items) for items in zip(*(coordinates(tracker,p) for p in value.split(32)),strict=True))
    def infer(y,q):
        series=[band_support(rank.clean_detail,a,b,bands,rank.selectors)
                for a,b in zip(y.split(48),q.split(48),strict=True)]
        return tuple(torch.cat(item) for item in zip(*series,strict=True))
    frames=track(cal)
    cut=infer(frames[0],frames[2])[0].amax(0)+1e-5
    root=Path(rt['source_root'])
    raw,rx,ids,_=source_role(root,contract,'U_s')
    _,_,ix=partition(rx)
    take=np.concatenate([x[:30] for x in np.split(ix,len(SOURCE))])
    def decompose(spectrum):
        common=spectrum.mean(1,keepdim=True)
        return {'common':common,'private':spectrum-common}
    def compare(test,ref):
        p=ref.abs().square().sum((1,2))
        c=(test*ref.conj()).sum((1,2))
        valid=p>1e-12
        error=(test-ref).abs().square().sum((1,2))
        return dict(relative_error=float(error.sum()/p.sum().clamp_min(1e-12)),
                    median_gain=float((c.real[valid]/p[valid]).median()),
                    median_phase_rad=float(c[valid].angle().abs().median()))
    training=torch.fft.fft(track(fit)[0],norm='ortho')
    private_training=decompose(training)['private']
    fit_groups=private_training.reshape(len(SOURCE),-1,*private_training.shape[1:])
    cohort_rx=torch.arange(len(SOURCE)).repeat_interleave(30)
    low=bands.bool()
    bases={}
    def pack(w):
        return torch.view_as_real(w.contiguous()).flatten(1)
    def unpack(w, template):
        return torch.view_as_complex(w.reshape(*template.shape,2).contiguous())
    for column in range(len(low)):
        for receiver_index in range(len(SOURCE)):
            # Neither the held receiver's clean input nor an exact replay trace fits this span.
            other=fit_groups[torch.arange(len(SOURCE))!=receiver_index].flatten(0,1)[...,low[column]]
            data=pack(other).double()
            data=data/data.norm(dim=1,keepdim=True).clamp_min(1e-6)
            _,axes=torch.linalg.eigh(data.T@data)
            bases[(column,receiver_index)]=axes[:,-16:].flip(1).float()
    output={}
    with ProcessPoolExecutor(max_workers=6) as pool,torch.no_grad():
        for route in ('full','residual'):
            for scene in (VIEWS[3],VIEWS[-1]):
                inp=((route,scene,raw[i],rx[i],ids[i]) for i in take)
                received,signal,clean=(tensor(np.stack(v),'cpu') for v in
                                     zip(*pool.map(replay,inp,chunksize=6),strict=True))
                y,rot,q,r=track(received)
                true=complex_iq(signal)/(rot*r)
                source_clean=track(clean)[0]
                reference={k:decompose(torch.fft.fft(w,norm='ortho'))
                           for k,w in (('matched_noise_free',true),('matched_source_clean_shape',source_clean))}
                estimate=torch.cat([rank(a,b,c)[-1] for a,b,c in zip(y.split(16),q.split(16),rot.split(16),strict=True)])
                original=torch.fft.fft(y,norm='ortho')
                edit=torch.fft.fft(estimate-y,norm='ortho')
                active=corruption_score(y)>old_threshold
                gain,optimum=infer(y,q)
                legacy=(.5*active[:,None]).float().expand(len(active),4)
                _,taper,boost=band_doses(active,gain,optimum,cut)
                band_names=('1_to_8','9_to_32','33_to_64','65_to_128')
                taper_increment=edit*(taper@bands)[:,None]
                boost_increment=edit*((boost-taper)@bands)[:,None]
                base=original+taper_increment
                parts=decompose(boost_increment)
                rows={'taper':base, 'boost':base+boost_increment,
                      'only_common_boost':base+parts['common'],
                      'only_private_boost':base+parts['private']}
                matched_private=reference['matched_noise_free']['private']
                correction=torch.zeros_like(base)
                for selector in bands.bool():
                    ref=matched_private[...,selector]
                    test=parts['private'][...,selector]
                    clean_power=ref.abs().square().sum((1,2),keepdim=True)
                    projection=(test*ref.conj()).sum((1,2),keepdim=True).real
                    correction[...,selector]=ref*(projection/clean_power.clamp_min(1e-10)).clamp(max=0.)
                rows['known_private_no_negative_gain']=rows['boost']-correction
                invariant=decompose(rows['known_private_no_negative_gain']-rows['boost'])
                if invariant['common'].abs().amax()>1e-7:
                    raise ValueError('Private-only projection changed the shared edit')
                observed_private=decompose(original)['private']
                for rank_label,rank_count in (('raw_private_ray',0),('source_private_span4',4),
                                               ('source_private_span16',16)):
                    stop=torch.zeros_like(base)
                    for column,selector in enumerate(low):
                        private=observed_private[...,selector]
                        direction=pack(private)
                        if rank_count:
                            projected=torch.empty_like(direction)
                            for receiver_index in range(len(SOURCE)):
                                chosen=cohort_rx==receiver_index
                                basis=bases[(column,receiver_index)][:,:rank_count]
                                projected[chosen]=(direction[chosen] @ basis) @ basis.T
                            direction=projected
                        template=unpack(direction,private)
                        proj=(parts['private'][...,selector]*template.conj()).sum((1,2),keepdim=True).real
                        power=template.abs().square().sum((1,2),keepdim=True)
                        stop[...,selector]=template*(proj/power.clamp_min(1e-10)).clamp(max=0.)
                    rows[rank_label]=rows['boost']-stop
                results={}
                for label,wave in rows.items():
                    observed=decompose(wave)
                    results[label]={
                        ref:{component:{name:compare(observed[component][...,selector],value[...,selector])
                                        for name,selector in zip(band_names,bands.bool(),strict=True)}
                             for component,value in source.items()}
                        for ref,source in reference.items()}
                results['increment_gain_affected'] = {}
                for name,selector in zip(band_names,bands.bool(),strict=True):
                    p=parts['private'][...,selector]
                    r=matched_private[...,selector]
                    projection=(p*r.conj()).sum((1,2)).real
                    energy=p.abs().square().sum((1,2))
                    mask=energy>1e-9
                    results['increment_gain_affected'][name]=dict(
                        observed=int(mask.sum()), negative=int((mask&(projection<0)).sum()),
                        negative_fraction_of_private_edit_power=float(
                            correction[...,selector].abs().square().sum()/energy.sum().clamp_min(1e-10)))
                output.setdefault(route,{})[scene]=results
    print(json.dumps(dict(n=len(take),source_role='held_U_s',target_reads=0,F_reads=0,V_reads=0,TX_reads=0,
                          actual_noise_measurement_after_freeze=True, oracle_diagnostic_only=True, deployable_approximation_fit_clean_only=True, train_leave_one_receiver_out=True, fit_per_receiver=int(fit_groups.shape[1]), by_route=output)),flush=True)

if __name__=='__main__':
    with torch.no_grad():
        main()
