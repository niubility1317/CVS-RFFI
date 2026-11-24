"""Isolated source diagnostic of independent noise observations at fixed LoRa action."""

import numpy as np

from .packet_physics import channel,chirp_samples,pack_windows,residual_channel
from .physical_inverse import full_action,residual_actions,solve_observations
from .registered_inverse import projected_operator,sampling_matrix


DRAWS = (2,4,8,25)
INVERSE_DRAWS = (8,25)
SEED = 2026114951


def receiver_noise(main,image,variance,rng):
    draw = rng.normal(size=(len(main),2)) * np.sqrt(variance/2.)
    thermal=draw[:,0]+1j*draw[:,1]
    return main*thermal+image*thermal.conj()


def cumulative_observations(raw,signal,action,variance,rng,sampling=lambda x:x):
    """Sums of fresh Gaussian innovations through the same linear noise path."""
    total=raw-signal
    series=[]
    previous=1
    for count in DRAWS:
        increment=receiver_noise(action.main,action.image,variance*(count-previous),rng)
        total=total+sampling(increment)
        series.append(signal+total/count)
        previous=count
    return tuple(series)


def pack_relative(wave,scale):
    return np.stack((wave.real,wave.imag),axis=-2).astype(np.float32)/scale


def full_replicas(packet,scene,receiver,pid,rng):
    action,source,capture,noiseless,_,gain,offset,error=full_action(packet,scene,receiver,pid)
    sampler=sampling_matrix(len(capture),offset)
    current=chirp_samples(capture,offset)
    observed=current.astype(np.complex128).ravel()/gain
    truth=sampler@noiseless
    check=float(np.linalg.norm(sampler@action(source)-truth)/max(np.linalg.norm(truth),1e-12))
    if not np.isfinite(check) or check>2e-9:
        raise ValueError('Projected operator does not reproduce the fixed noiseless source example')
    variance=10**(-channel(scene).reference_snr_db/10)
    samples=cumulative_observations(observed,truth,action,variance,rng,sampling=sampler.__matmul__)
    damp=np.sqrt(3*np.mean(np.abs(observed-truth)**2))
    inverted=solve_observations(projected_operator(action,sampler),
                                tuple(samples[DRAWS.index(k)] for k in INVERSE_DRAWS),damp)
    scale=np.float32(np.sqrt(np.mean(np.abs(current)**2)))
    normalized=tuple(pack_relative((s*gain).reshape(current.shape),scale) for s in samples)
    clean=pack_relative((truth*gain).reshape(current.shape),scale)
    return pack_windows(current),clean,normalized,tuple(pack_windows(chirp_samples(x)) for x in inverted),max(error,check)


def residual_replicas(packet,scene,receiver,pid,rng):
    variance=10**(-residual_channel(scene).reference_snr_db/10)
    records=list(residual_actions(packet,scene,receiver,pid))
    capture=np.stack([v[1] for v in records])
    scale=np.float32(np.sqrt(np.mean(np.abs(capture)**2)))
    by_chirp,inverts=[],[]
    for action,observed,signal,_,gain,_ in records:
        raw=observed.astype(np.complex128)/gain
        wave=cumulative_observations(raw,signal,action,variance,rng)
        by_chirp.append([x*gain for x in wave])
        damp=np.sqrt(3*np.mean(np.abs(raw-signal)**2))
        estimate=solve_observations(action.operator(),tuple(wave[DRAWS.index(k)] for k in INVERSE_DRAWS),damp)
        inverts.append([x[32:288] for x in estimate])
    received=tuple(pack_relative(np.stack(x),scale) for x in zip(*by_chirp,strict=True))
    reconstructed=tuple(pack_windows(np.stack(x)) for x in zip(*inverts,strict=True))
    clean=pack_relative(np.stack([row[2]*row[4] for row in records]),scale)
    return pack_windows(capture),clean,received,reconstructed,max(row[-1] for row in records)


def replicate(route,scene,row):
    from leo_practical.channel import stable_seed
    packet,receiver,pid=row
    rng=np.random.default_rng(stable_seed(SEED,route,scene,receiver,str(pid)))
    solve=full_replicas if route=='full' else residual_replicas
    return solve(packet,scene,receiver,pid,rng)
