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

"""Detect diffuse weak disturbances from disjoint receiver-only spectral evidence."""

import math

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit



class WeakDisturbanceFeatures(torch.nn.Module):
    """Jointly observe physical guard bands and dechirped repeat innovations."""

    def __init__(self):
        super().__init__()
        frequency = (256 * torch.fft.fftfreq(256)).round().abs()
        self.register_buffer('window', torch.hann_window(256))
        self.register_buffer('guards', torch.stack(((frequency >= 70) & (frequency <= 91),
                                                   (frequency >= 92) & (frequency <= 119))))
        self.register_buffer('innovations', torch.stack(((frequency >= 12) & (frequency <= 32),
                                                        (frequency >= 33) & (frequency <= 64),
                                                        (frequency >= 65) & (frequency <= 104))))

    @staticmethod
    def summaries(power, masks):
        """A robust floor and flatness for each band, then two packet order statistics."""
        summaries = []
        for mask in masks:
            band = power[..., mask]
            floor = band.median(-1).values.clamp_min(1e-12)
            total = band.mean(-1).clamp_min(1e-12)
            for scalar in (floor.log(), (floor/total).log()):
                ordered = scalar.sort(dim=1).values
                summaries.extend((ordered[:, 2], ordered[:, 4]))
        return summaries

    def forward(self, iq, observed, quality):
        raw = complex_iq(iq)
        raw_rms = raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        raw = (raw / raw_rms) * self.window / self.window.square().mean().sqrt()
        guard_power = torch.fft.fft(raw, norm='ortho').abs().square()
        spec = torch.fft.fft(observed, norm='ortho')
        weight = quality.square().clamp_min(1e-6)
        repeat_power = (spec - (spec * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True)).abs().square()
        # The receiver observes gradual floors and dispersion, not transmitter identity or receiver ID.
        a = self.summaries(guard_power, self.guards)
        b = self.summaries(repeat_power, self.innovations)
        return torch.stack(a+b, dim=1)


def monotone_scores(positive_inputs, negative_inputs):
    """Use paired generic corruptions for clean-versus-impaired logistic fitting."""
    midpoint = negative_inputs.median(dim=0).values
    p25,p75 = torch.quantile(negative_inputs, negative_inputs.new_tensor([.25,.75]),dim=0)
    dispersion = (p75-p25).clamp_min(.10)
    a = ((negative_inputs-midpoint)/dispersion).clamp(-10,10).double().numpy()
    b = ((positive_inputs-midpoint)/dispersion).clamp(-10,10).double().numpy()
    values = np.concatenate((a,b))
    target = np.concatenate((np.zeros(len(a)),np.ones(len(b))))
    weights = np.concatenate((np.full(len(a),.5/len(a)),np.full(len(b),.5/len(b))))
    def loss(state):
        w = state[:-1]
        z = values@w + state[-1]
        objective = np.sum(weights*(np.logaddexp(0,z)-target*z)) + .025 * np.dot(w,w)
        delta = weights*(expit(z)-target)
        gradient = np.concatenate((values.T@delta+.05*w,[delta.sum()]))
        return objective,gradient
    result=minimize(loss,np.zeros(values.shape[1]+1),method='L-BFGS-B',jac=True,
                    bounds=[(0,None)]*values.shape[1]+[(None,None)],options=dict(maxiter=150))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Generic weak-disturbance fit failed: {result.message}')
    return midpoint,dispersion,negative_inputs.new_tensor(result.x[:-1]),float(result.x[-1])


def apply_scores(row, midpoint, dispersion, weight, bias):
    return ((row-midpoint)/dispersion).clamp(-10,10) @ weight + bias


def generic_noise(iq, rng, amplitude=None):
    """Generic independent complex noise, either flat or smoothly colored."""
    size=iq.shape
    opts=dict(device=iq.device,generator=rng)
    draw=torch.complex(torch.randn(size,**opts),torch.randn(size,**opts))
    frequency=torch.fft.fftfreq(size[-1],device=iq.device)
    angle=2*torch.pi*(frequency-torch.rand((len(iq),1,1),**opts))
    depth=.95*torch.rand((len(iq),1,1),**opts)
    flat=torch.rand((len(iq),1,1),**opts)<.40
    profile=(1+torch.where(flat,0.,depth)*angle.cos()).sqrt()
    draw=torch.fft.ifft(torch.fft.fft(draw) * profile)
    if amplitude is None:
        amplitude=math.exp(math.log(.006))*torch.exp(math.log(.16/.006)*torch.rand((len(iq),1,1),**opts))
    return iq + amplitude * draw/2**.5

def cases(y, signal):
    added = complex_iq(y-signal)
    frequency = (256*torch.fft.fftfreq(added.shape[-1])).round().abs()
    gain = (.5+.5*torch.cos(torch.pi*((frequency-60)/8).clamp(0,1)))
    spectrum = torch.fft.fft(added,norm='ortho')
    filtered = torch.fft.ifft(gain*spectrum,norm='ortho')
    relative_filter_energy = (filtered.abs().square().sum()/added.abs().square().sum()).item()
    # Unit response throughout 0–60 kHz; no rescaling or gate retuning.
    check = torch.fft.fft(filtered,norm='ortho')[...,frequency<=60]-spectrum[...,frequency<=60]
    assert check.abs().max()<=2e-6
    return {'registered_received':y,'noise_restricted_to_occupied_band':signal+real_iq(filtered),
            'exact_added_noise_removed':signal}, relative_filter_energy

def main():
    rt = read(CURRENT/'runtime.json')
    activate(rt)
    sys.path.insert(0,str(EXACT/'code'))
    from leo_practical.channel import ChannelStream
    import inspect
    assert 'return_operator' in inspect.signature(ChannelStream.process).parameters
    torch.set_num_threads(2)
    source=Path(rt['source_root'])
    contract=source_contract(source)
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held=partition(rx)
    samples=np.concatenate([rows[:45] for rows in np.split(held,len(SOURCE))])
    # Coefficients and threshold were determined without the actual practical replays.
    settings=json.loads(r'''{"midpoint":[-10.469514846801758,-10.252159118652344,-0.4258999824523926,-0.2927226722240448,-10.231888771057129,-10.041860580444336,-0.7853565216064453,-0.5574744939804077],"dispersion":[0.4775829315185547,0.4728384017944336,0.23064222931861877,0.19363337755203247,0.4351682662963867,0.43538951873779297,0.3553733825683594,0.3032861649990082],"coef":[0.5706962943077087,0.5864372253417969,0.1895655244588852,0.14165742695331573,0.586089551448822,0.5927881002426147,0.1883552372455597,0.16225562989711761],"offset":-3.0606591959790537,"threshold":0.09975529670715332}''')
    fits=tuple(torch.tensor(settings[key],dtype=torch.float32) for key in ('midpoint','dispersion','coef'))
    cut=settings['threshold']
    old=0.045941156521439555
    tracker=PacketRestorer('consensus').eval()
    features=WeakDisturbanceFeatures()
    stats={}
    with ProcessPoolExecutor(max_workers=6) as pool,torch.no_grad():
        for route in ('full','residual'):
            for scene in (VIEWS[1],VIEWS[3],VIEWS[-1]):
                items=((route,scene,raw[i],rx[i],ids[i]) for i in samples)
                pairs=list(pool.map(replay,items,chunksize=8))
                iq, signal, _ = (tensor(np.stack(rows),'cpu') for rows in zip(*pairs,strict=True))
                waves,energy=cases(iq,signal)
                current={}
                for kind,v in waves.items():
                    y,p,q,r=coordinates(tracker,v)
                    raw_features=features(v,y,q)[:,:8]
                    score=apply_scores(raw_features,*fits,settings['offset'])
                    outer=corruption_score(y)
                    current[kind]=dict(guard=int((score>cut).sum()),outer=int((outer>old).sum()),
                                       median_score=float(score.median()),median_outer=float(outer.median()))
                stats.setdefault(route,{})[scene]=dict(relative_noise_energy_after_filter=energy, cases=current)
    print(json.dumps(dict(n=len(samples),threshold=cut,old_threshold=old,passband_khz=60,stopband_khz=68,
                      source_role='held_U_s',actual_noise_forensic_only=True,
                      target_reads=0,F_reads=0,V_reads=0,TX_reads=0,by_route=stats)),flush=True)

if __name__=='__main__':
    main()
