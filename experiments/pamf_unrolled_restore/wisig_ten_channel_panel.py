"""Ten independent source-IQ simulation strata and disjoint V observations."""

from pathlib import Path
import sys

import numpy as np

from . import wisig_multichannel_panel as earlier
from . import wisig_multichannel_guard_audit as guard
from .wisig_independent_channel_audit import reflect


EXTRA = ('fractional_clock','accelerating_doppler','time_selective','narrowband')
FAMILIES = tuple(family for family,_ in earlier.DEVELOPMENT)+EXTRA
LABELS = earlier.TEST_LABELS+tuple(f'{family}/{view}' for family in EXTRA for view in ('nominal','stress'))
NDEV = 48
NTEST = 60
SEED = 2026112506


def split(seed):
    used = np.r_[earlier.prior_audit.select(),earlier.prior_opt.selected(),earlier.holdout(),
                 earlier.fresh_holdout(guard.SEED)]
    selected = []
    for start in range(0,360,24):
        eligible = np.setdiff1d(np.arange(start,start+24),used)
        order = np.random.default_rng(SEED+start).permutation(eligible)
        if len(order)!=8 or seed not in (SEED,SEED+1):
            raise ValueError('Only two disjoint 60-packet diagnostic panels are available')
        selected.extend(np.sort(order[4*(seed-SEED):4*(seed-SEED)+4]))
    return np.asarray(selected)


def interp(x,coordinates):
    floor = np.floor(coordinates).astype(int)
    support = floor[...,None]+np.arange(-5,7)
    distance = coordinates[...,None]-support
    weight = np.sinc(distance)*np.sinc(distance/6.)
    return np.sum(x[reflect(support)]*weight,axis=-1)/weight.sum(-1)


def extra_channel(clean,family,view,seed,reference=False):
    rng = np.random.default_rng(seed)
    n,tcount = clean.shape
    t = np.arange(tcount)
    u = np.linspace(-1.,1.,tcount)
    stress = view=='stress'
    action = np.empty_like(clean,dtype=np.complex128)
    for i,x in enumerate(clean):
        count = 3 if family=='fractional_clock' else int(rng.integers(2,5))
        lag = int(rng.integers(-9,10))+np.r_[0,np.sort(rng.choice(np.arange(2,26),count-1,replace=False))]
        gain = np.r_[1.,rng.uniform(.25,.7,count-1)]*np.exp(1j*rng.uniform(-np.pi,np.pi,count))
        rate = rng.uniform(-1.7,1.7)
        phase = rate*u
        if family=='fractional_clock':
            lag = lag+rng.uniform(-.46,.46,count)
            stretch = rng.uniform(-1.,1.)*(.8 if stress else .32)
            paths = interp(x,t[None,:]-lag[:,None]-stretch*u)
        else:
            paths = x[reflect(t[None,:]-lag[:,None])]
        if family=='accelerating_doppler':
            phase += rng.choice([-1.,1.])*rng.uniform(.35,2.2 if stress else 1.1)*u**2
        if family=='time_selective':
            slopes = rng.uniform(-.7,.7,count)*(1.4 if stress else .7)
            motion = rng.uniform(-1.,1.,count)*(1.6 if stress else .75)
            paths = paths*(1+slopes[:,None]*u)*np.exp(1j*motion[:,None]*u)
        action[i] = (gain@paths)*np.exp(1j*phase)
    v = np.mean(np.abs(action)**2,axis=1)*10**(-rng.uniform(9.,24.,n)/10.)
    noise = earlier.gaussian(rng,action.shape)*np.sqrt(v[:,None])
    if family=='narrowband':
        for _ in range(2 if stress else 1):
            frequency = rng.uniform(-.49,.49,n)
            phase = rng.uniform(-np.pi,np.pi,n)
            power = v*rng.uniform(.5,2.4 if stress else 1.4,n)
            noise += np.sqrt(power[:,None])*np.exp(1j*(2*np.pi*frequency[:,None]*t+phase[:,None]))
            v = v+power
    received = action+noise
    return (received,action,v) if reference else received


def development(root,source,frozen):
    with np.load(frozen,allow_pickle=False) as data:
        x,y = list(data['clean'][:,:NDEV]),list(data['received'][:,:NDEV])
    clean,_ = earlier.prior_opt.dev_pairs(source)
    rng = np.random.default_rng(SEED)
    for i,family in enumerate(EXTRA):
        samples = clean[rng.choice(len(clean),NDEV,replace=False)]
        x.append(samples)
        y.append(extra_channel(samples,family,'nominal',SEED+400+i))
    np.savez(root/'development.npz',clean=x,received=y)
    print('DEVELOPMENT_IQ',len(x),NDEV,flush=True)


def challenge(root,panel,seed):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
    take = split(seed)
    with np.load(panel/'scorer_reference.npz',allow_pickle=False) as points:
        clean = points['clean'][take]
        signal = list(points['signal'][:,take])
        variance = list(points['variance'][:,take])
    base = np.load(panel/'received.npy',mmap_mode='r',allow_pickle=False)
    observed = list(earlier.paired.voltages(base[:,take]))
    generator_cases = (*earlier.ADDITIONAL_TEST, *((family,view) for family in EXTRA for view in ('nominal','stress')))
    for i,(family,view) in enumerate(generator_cases):
        generator = extra_channel if family in EXTRA else earlier.simulate
        y,h,v = generator(clean,family,view,seed+1000+i,reference=True)
        observed.append(y)
        signal.append(h)
        variance.append(v)
        print('HOLDOUT_IQ',family,view,len(clean),flush=True)
    np.savez(root/'challenge.npz',clean=clean,received=observed)
    np.savez(root/'oracle.npz',signal=signal,variance=variance)
