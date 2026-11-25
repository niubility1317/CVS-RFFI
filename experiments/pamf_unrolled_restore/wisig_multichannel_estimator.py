"""IQ-only selection between compact and sparse, rotating receive actions."""

import numpy as np

from . import wisig_paired_estimator as compact
from .wisig_independent_channel_audit import FIT, N, TEST, reflect


LAGS = np.arange(-46,47)
SPLITS = np.arange(len(FIT))//8 % 3
MODELS = (('compact',0), ('sparse',4), ('sparse',8), ('sparse',12),
          ('rotating',8), ('rotating',12), ('mirror',8), ('mirror',12))
SIZES = np.array([9,4,8,12,9,13,9,13])
COMPLEXITY = (0., .5, 2.)
NOISE_GRID = ((0.,0.),(.5,0.),(1.,0.),(.5,.5),(1.,.5))


def basis(clean,times,mirror):
    direct = clean[reflect(times[:,None]-LAGS)]
    return np.column_stack((direct,direct.conj())) if mirror else direct


def action(clean,coefficient,omega,times):
    return (basis(clean,times,True) @ coefficient)*np.exp(1j*omega*times)


def ridge(matrix,observed):
    gram = matrix.conj().T @ matrix
    ridge_scale = max(float(np.trace(gram).real/len(gram)),1e-18)*.002
    return np.linalg.solve(gram+ridge_scale*np.eye(len(gram)),matrix.conj().T@observed)


def pursuit(matrix,observed,count):
    chosen = []
    norm = np.linalg.norm(matrix,axis=0).clip(1e-12)
    residual = observed.copy()
    for _ in range(count):
        peak = np.abs(matrix.conj().T@residual)/norm
        peak[chosen] = -np.inf
        chosen.append(int(np.argmax(peak)))
        fitted = matrix[:,chosen]
        coefficient = ridge(fitted,observed)
        residual = observed-fitted@coefficient
    return chosen,coefficient


def drift(wanted,proposed,times):
    """Fit a common phase slope on local, power-weighted cross-products."""
    block = times//8
    unique, b = np.unique(block,return_inverse=True)
    products = wanted*proposed.conj()
    pooled = (np.bincount(b,weights=products.real)+1j*np.bincount(b,weights=products.imag))
    weights = np.abs(pooled).clip(1e-16)
    time = 8*unique+3.5
    centered = time-np.average(time,weights=weights)
    slope = np.sum(weights*centered*np.unwrap(np.angle(pooled)))/np.sum(weights*centered**2).clip(1e-16)
    return float(np.clip(slope,-.035,.035))


def embed(delay,taps):
    coeff = np.zeros(2*len(LAGS),complex)
    coeff[delay+compact.OFFSETS-LAGS[0]] = taps
    return coeff


def fit_compact(clean,obs,prior,keep):
    times = FIT[keep]
    candidates = clean[reflect(times[None,:]-compact.DELAYS[:,None])]
    y = obs[times]
    score = np.abs(candidates.conj()@y)**2/np.sum(np.abs(candidates)**2,axis=-1).clip(1e-16)
    ids = np.unique(np.r_[np.argsort(score)[-6:],np.argsort(prior)[-3:]])
    delays = compact.DELAYS[ids]
    bank = clean[reflect(times[None,:,None]-delays[:,None,None]-compact.OFFSETS)]
    candidates = []
    for n in (5,9,13):
        radius = n//2
        subset = bank[...,6-radius:7+radius]
        fit = compact.coefficients(subset,y)
        error = np.mean(np.abs(np.einsum('mtk,mk->mt',subset,fit)-y)**2,axis=-1)
        best = int(np.argmin(error*(1+2*n/len(y))))
        candidates.append((error[best]*(1+2*n/len(y)), delays[best],
                           np.pad(fit[best],(6-radius,6-radius))))
    _, delay, coeff = min(candidates,key=lambda item:item[0])
    return embed(delay,coeff),0.


def fit_model(clean,obs,kind,count,prior,keep):
    if kind=='compact':
        return fit_compact(clean,obs,prior,keep)
    times = FIT[keep]
    y = obs[times]
    direct = basis(clean,times,False)
    omega = 0.
    if kind in ('rotating','mirror'):
        columns,taps = pursuit(direct,y,4)
        omega = drift(y,direct[:,columns]@taps,times)
    matrix = basis(clean,times,kind=='mirror')
    columns,taps = pursuit(matrix,y*np.exp(-1j*omega*times),count)
    coefficients = np.zeros(2*len(LAGS),complex)
    coefficients[columns] = taps
    return coefficients,omega


def pack(clean,obs,prior):
    errors, records = [],[]
    for model in MODELS:
        residual = np.empty(len(FIT),complex)
        for fold in range(3):
            held = SPLITS==fold
            coeff,omega = fit_model(clean,obs,*model,prior,~held)
            residual[held] = obs[FIT[held]]-action(clean,coeff,omega,FIT[held])
        coeff,omega = fit_model(clean,obs,*model,prior,np.ones(len(FIT),bool))
        errors.append(float(np.mean(np.abs(residual)**2)))
        records.append((coeff,omega,residual))
    return np.asarray(errors),records


def choose(bank,penalty):
    error,records = bank
    index = int(np.argmin(error*(1+penalty*SIZES/len(FIT))))
    return index,records[index]


def quiet_spectrum(clean,obs):
    x = np.fft.fft(clean[None,:]*compact.TAPERS,axis=-1,norm='ortho')
    y = np.fft.fft(obs[None,:]*compact.TAPERS,axis=-1,norm='ortho')
    quiet = np.argsort(np.mean(np.abs(x)**2,axis=0))[:48]
    return quiet,np.mean(np.abs(y)**2,axis=0)[quiet]


def noise(spectrum,residual,color,burst):
    quiet,power = spectrum
    adjacent = np.diff(FIT)==1
    a,b = residual[:-1][adjacent],residual[1:][adjacent]
    rho = np.sum((a.conj()*b).real)/np.sum(np.abs(a)**2).clip(1e-18)
    rho = np.clip(rho,-.85,.85)*color
    shape = (1-rho*rho)/(1+rho*rho-2*rho*np.cos(2*np.pi*quiet/N))
    colored = float(np.median(power/shape)/.891353)
    power = np.abs(residual)**2
    bulk = np.median(power)/np.log(2.)
    excess = max(0.,float(power.mean()-1.7*bulk))
    return colored+burst*excess


def tune(groups):
    """Pick global complexity and coloring from learned-family IQ holdouts."""
    prior = compact.learn_delay_prior(np.concatenate([x[:len(x)//2] for x,y in groups]),
                                      np.concatenate([y[:len(y)//2] for x,y in groups]))
    prepared = []
    for clean,obs in groups:
        score = []
        for x,y in zip(clean[len(clean)//2:],obs[len(obs)//2:]):
            score.append((x,y,pack(x,y,prior),quiet_spectrum(x,y)))
        prepared.append(score)
    choice = []
    for penalty in COMPLEXITY:
        family_error = []
        for rows in prepared:
            errors = []
            for x,y,bank,_ in rows:
                _,(coef,omega,_) = choose(bank,penalty)
                err = np.mean(np.abs(y[TEST]-action(x,coef,omega,TEST))**2)
                errors.append(err/np.mean(np.abs(y[TEST])**2).clip(1e-18))
            family_error.append(float(np.quantile(errors,.5)+np.quantile(errors,.9)))
        choice.append(dict(penalty=penalty,criterion=float(np.mean(family_error)),per_group=family_error))
    fixed = min(choice,key=lambda item:item['criterion'])['penalty']
    noise_choice = []
    for color,burst in NOISE_GRID:
        groups_loss = []
        for rows in prepared:
            values = []
            for x,y,bank,spectrum in rows:
                _,(coef,omega,residual) = choose(bank,fixed)
                v0 = max(1e-12,noise(spectrum,residual,0.,0.))
                v = max(1e-12,noise(spectrum,residual,color,burst))
                mse = np.mean(np.abs(y[TEST]-action(x,coef,omega,TEST))**2)
                values.append(mse/v-mse/v0+np.log(v/v0))
            groups_loss.append(float(np.median(values)))
        noise_choice.append(dict(color=color,burst=burst,criterion=float(np.mean(groups_loss)),per_group=groups_loss))
    selected = min(noise_choice,key=lambda item:item['criterion'])
    return dict(prior=prior.tolist(),penalty=fixed,color=selected['color'],burst=selected['burst'],
                action_grid=choice,noise_grid=noise_choice)


def estimate(clean,obs,config):
    prior = np.asarray(config['prior'])
    rows = []
    for x,y in zip(clean,obs):
        index,(coeff,omega,residual) = choose(pack(x,y,prior),config['penalty'])
        variance = noise(quiet_spectrum(x,y),residual,config['color'],config['burst'])
        rows.append((coeff,omega,variance,index))
    return tuple(map(np.asarray,zip(*rows)))
