"""Exact source-trace inverse restricted to the registered LoRa observations."""

from functools import lru_cache

import numpy as np
from scipy.signal import resample_poly
from scipy.sparse import csr_matrix

from .data import WINDOW_START
from .packet_data import CHIRPS
from .packet_physics import chirp_samples, pack_windows
from .physical_inverse import full_action, real_operator, residual_actions, solve_observations


@lru_cache(maxsize=None)
def sampling_matrix(length, offset):
    """Sparse linear map from native samples to six registered chirps."""
    local = resample_poly(np.eye(1250), up=32, down=125, axis=-1)[:, 32:288].T
    row, column = np.nonzero(local)
    starts = np.asarray(CHIRPS)*1024+WINDOW_START-125+offset
    if starts[0]<0 or starts[-1]+1250>length:
        raise ValueError('Registered sampling needs the native reception context')
    output = (256*np.arange(len(CHIRPS))[:, None]+row).ravel()
    native = (starts[:, None]+column).ravel()
    value = np.tile(local[row,column], len(CHIRPS))
    return csr_matrix((value,(output,native)), shape=(256*len(CHIRPS),length))


def projected_operator(action, sample):
    adjoint = sample.T.tocsr()
    return real_operator(lambda x: sample@action(x),
                         lambda y: action.adjoint_iq(adjoint@y),
                         (sample.shape[0],action.propagation.shape[1]))


def solve_reads(operator, raw, signal, noise_rms, learned):
    known_80 = signal+.20*(raw-signal)
    inputs = (raw, raw-learned, raw-.5*learned, known_80, signal)
    return solve_observations(operator, inputs, np.sqrt(3.)*noise_rms)


def inverse_full(packet, scene, rx, pid, registered, noise_guess, rms_guess):
    action, source, received, signal, _, gain, offset, error = full_action(packet,scene,rx,pid)
    sampling = sampling_matrix(len(received),offset)
    aligned = chirp_samples(received,offset)
    if not np.array_equal(registered,pack_windows(aligned)):
        raise ValueError('Projected full inverse changed the registered observation')
    observed = np.asarray(aligned,dtype=np.complex128).ravel()/gain
    noiseless = sampling@signal
    check = float(np.linalg.norm(sampling@action(source)-noiseless)/max(np.linalg.norm(noiseless),1e-12))
    if max(check,error)>2e-9:
        raise ValueError('Full source action does not preserve registered coordinates')
    unit = np.sqrt(np.mean(np.abs(aligned)**2))/gain
    noise_rms = np.sqrt(np.mean(rms_guess**2))*unit
    result = solve_reads(projected_operator(action,sampling),observed,noiseless,
                         noise_rms,noise_guess.ravel()*unit)
    return np.stack([pack_windows(chirp_samples(value)) for value in result]), max(error,check)


def inverse_residual(packet, scene, rx, pid, registered, noise_guess, rms_guess):
    records = list(residual_actions(packet,scene,rx,pid))
    observed = np.stack([row[1] for row in records])
    if not np.array_equal(registered,pack_windows(observed)):
        raise ValueError('Residual inverse changed the registered observation')
    unit = np.sqrt(np.mean(np.abs(observed)**2))
    estimates, errors = [], []
    for (action,obs,signal,_,gain,error), learned, std in zip(records,noise_guess,rms_guess,strict=True):
        result = solve_reads(action.operator(),obs.astype(np.complex128)/gain,signal,std*(unit/gain),
                             learned*(unit/gain))
        estimates.append([v[32:288] for v in result])
        errors.append(error)
    return np.stack([pack_windows(np.stack(v)) for v in zip(*estimates,strict=True)]), max(errors)


def inverse_row(route, scene, row):
    packet, rx, pid, registered, noise, rms = row
    estimate = noise[:,0]+1j*noise[:,1]
    solve = inverse_full if route=='full' else inverse_residual
    return solve(packet,scene,rx,pid,registered,estimate,rms)
