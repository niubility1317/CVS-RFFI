"""Prepare source-only WiSig IQ across independent simulator families."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from . import wisig_independent_channel_audit as prior_audit
from . import wisig_paired_estimator_audit as prior_opt
from . import wisig_paired_estimator as paired
from .wisig_independent_channel_audit import reflect


SEED = 2026112504
NDEV = 80
NTEST = 60
DEVELOPMENT = (
    ('practical',()), ('weak_leo',('leo_clear_weak','leo_low_elev_weak')),
    ('legacy_sat',('clear_leo','rain_leo')),
    ('sparse_doppler',('sparse_nominal',)), ('image_ar',('colored_nominal',)),
    ('multipath_burst',('impulse_nominal',)),
)
ADDITIONAL_TEST = (
    ('weak_leo','leo_clear_weak'), ('weak_leo','leo_low_elev_weak'), ('weak_leo','leo_rain_weak'),
    ('legacy_sat','clear_leo'), ('legacy_sat','rain_leo'), ('legacy_sat','storm_mp'),
    ('sparse_doppler','sparse_nominal'), ('sparse_doppler','sparse_wide'),
    ('image_ar','colored_nominal'), ('image_ar','colored_reverse'),
    ('multipath_burst','impulse_nominal'), ('multipath_burst','impulse_heavy'),
)
TEST_LABELS = tuple(f'practical/{profile}/{route}/{scene}' for profile,route,scene in prior_audit.CELLS
                   ) + tuple(f'{family}/{view}' for family,view in ADDITIONAL_TEST)


def holdout():
    rng = np.random.default_rng(SEED)
    used = np.r_[prior_audit.select(), prior_opt.selected()]
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.arange(i,i+24),used),4,replace=False))
                           for i in range(0,360,24)])


def legacy(clean,name,seed,reference=True):
    import torch
    import sat_channel
    from training_controls import SAT_CHANNEL_SCENARIO_CONFIGS

    cfg = sat_channel.SatSimConfig(**SAT_CHANNEL_SCENARIO_CONFIGS[name])
    iq = torch.tensor(np.stack((clean.real,clean.imag),1).tolist(),dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    if not reference:
        received,_,_ = sat_channel.apply_sat_gnd_channel_batch(iq,cfg,gen=generator)
        return paired.voltages(np.asarray(received.tolist(),np.float32))

    add_noise = sat_channel.complex_awgn_like
    distort = sat_channel.apply_iq_imbalance
    trace = {}

    def record_noise(signal,snr_db,gen=None):
        trace['signal'] = signal
        trace['variance'] = signal.abs().square().mean(1).clamp_min(1e-12)*10**(-snr_db/10)
        return add_noise(signal,snr_db,gen)

    def record_image(observed,amp_db,phase_deg):
        trace['signal'] = distort(trace['signal'],amp_db,phase_deg)
        trace['variance'] = trace['variance']*(1+10**(amp_db/10))*.5
        return distort(observed,amp_db,phase_deg)

    with patch.object(sat_channel,'complex_awgn_like',record_noise), patch.object(
            sat_channel,'apply_iq_imbalance',record_image):
        received,_,_ = sat_channel.apply_sat_gnd_channel_batch(iq,cfg,gen=generator)
    received = np.asarray(received.tolist(),np.float32)
    signal = np.asarray(trace['signal'].tolist(),np.complex128)
    variance = np.asarray(trace['variance'].tolist(),np.float64)
    return paired.voltages(received),signal,variance


def gaussian(rng,shape):
    return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2.)


def stress(clean,view,seed,reference=True):
    rng = np.random.default_rng(seed)
    packets,length = clean.shape
    t = np.arange(length)
    step = t/(length-1)
    signal = np.empty_like(clean,dtype=np.complex128)
    colored = view.startswith('colored')
    bursts = view.startswith('impulse')
    wide = view == 'sparse_wide'
    for i,x in enumerate(clean):
        base = int(rng.integers(-8,10))
        count = int(rng.integers(2,7))
        extent = 29 if wide else 18
        extra = rng.choice(np.arange(2,extent),count-1,replace=False)
        delays = np.r_[base,base+np.sort(extra)]
        amplitude = np.r_[1.,np.exp(-np.arange(1,count)/4)*.62]
        taps = amplitude*np.exp(1j*rng.uniform(-np.pi,np.pi,count))
        movement = rng.uniform(-.33,.33,count) if wide else np.zeros(count)
        variation = np.exp(1j*movement[:,None]*step)
        waves = x[reflect(t[None,:]-delays[:,None])]
        propagated = (taps[:,None]*waves*variation).sum(0)
        if colored:
            propagated += rng.uniform(.08,.32)*np.exp(1j*rng.uniform(-np.pi,np.pi))*waves[0].conj()
        rotation = rng.uniform(-4.,4.) if wide else rng.uniform(-2.3,2.3)
        signal[i] = propagated*np.exp(1j*rotation*step)
    snr = rng.uniform(7.,25.,packets)
    variance = (np.abs(signal)**2).mean(-1)*10**(-snr/10)
    noise = gaussian(rng,signal.shape)
    if colored:
        lo,hi = (-.72,-.32) if view=='colored_reverse' else (.35,.78)
        rho = rng.uniform(lo,hi,packets)
        innovation = np.sqrt(1-rho*rho)
        for k in range(1,length):
            noise[:,k] = rho*noise[:,k-1]+innovation*noise[:,k]
    noise *= np.sqrt(variance[:,None])
    if bursts:
        probability = .09 if view=='impulse_heavy' else .035
        multiplier = 13. if view=='impulse_heavy' else 9.
        impulse = rng.random(signal.shape)<probability
        noise += np.sqrt(multiplier*variance[:,None])*gaussian(rng,signal.shape)*impulse
        variance *= 1.+probability*multiplier
    received = signal+noise
    return (received,signal,variance) if reference else received


def simulate(clean,family,view,seed,reference=True):
    if family in ('weak_leo','legacy_sat'):
        return legacy(clean,view,seed,reference)
    return stress(clean,view,seed,reference)


def prepare(root,source,panel):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
    clean,observed = prior_opt.dev_pairs(source)
    rng = np.random.default_rng(SEED)
    dev_x,dev_y = [],[]
    for i,(family,views) in enumerate(DEVELOPMENT):
        selected = rng.choice(len(clean),NDEV,replace=False)
        x = clean[selected]
        if views:
            chunks = np.array_split(x,len(views))
            y = np.concatenate([simulate(chunk,family,view,SEED+100*i+j,reference=False)
                                for j,(chunk,view) in enumerate(zip(chunks,views))])
            order = rng.permutation(NDEV)
            x,y = x[order],y[order]
        else:
            y = observed[selected]
        dev_x.append(x)
        dev_y.append(y)
        print('DEV_IQ',family,len(x),flush=True)
    # Development sees only anonymous, balanced groups of paired IQ.
    np.savez(root/'development.npz',clean=dev_x,received=dev_y)
    take = holdout()
    if len(take)!=NTEST:
        raise ValueError('Unexpected cross-family holdout count')
    received = np.load(panel/'received.npy',mmap_mode='r',allow_pickle=False)
    with np.load(panel/'scorer_reference.npz',allow_pickle=False) as past:
        x = past['clean'][take]
        old_signal = list(past['signal'][:,take])
        old_variance = list(past['variance'][:,take])
    y = list(paired.voltages(received[:,take]))
    for i,(family,view) in enumerate(ADDITIONAL_TEST):
        frame,signal,var = simulate(x,family,view,SEED+1000+i)
        y.append(frame)
        old_signal.append(signal)
        old_variance.append(var)
        print('HOLDOUT_IQ',family,view,len(x),flush=True)
    np.savez(root/'challenge.npz',clean=x,received=y)
    np.savez(root/'oracle.npz',signal=old_signal,variance=old_variance)
