"""Fit a compact receive operator and noise floor using IQ pairs only."""

import numpy as np

from .wisig_independent_channel_audit import DELAYS, FIT, N, TEST, reflect


OFFSETS = np.arange(-6, 7)
ORDERS = (3, 7, 13)
PRIOR_WEIGHTS = (0., 2., 8., 24.)
FOLDS = np.arange(len(FIT))//8 % 3
TAPERS = np.sqrt(2*N/(N+1)) * np.sin(
    np.pi*np.arange(1,4)[:,None]*np.arange(1,N+1)/(N+1))


def voltages(row):
    return row[...,0,:].astype(np.float64) + 1j*row[...,1,:]


def lag_scores(clean, observed):
    shifted = clean[reflect(FIT[None,:]-DELAYS[:,None])]
    y = observed[FIT]
    return np.abs(shifted.conj() @ y)**2 / (
        np.sum(np.abs(shifted)**2,axis=-1)*np.sum(np.abs(y)**2)).clip(1e-20)


def learn_delay_prior(clean, observed):
    scores = np.stack([lag_scores(x,y) for x,y in zip(clean,observed)])
    strength = scores.max(1)
    eligible = strength >= np.quantile(strength,.65)
    counts = np.bincount(scores[eligible].argmax(1),minlength=len(DELAYS))
    smoothing = np.exp(-.5*((DELAYS[:,None]-DELAYS)/1.5)**2)
    weights = smoothing @ counts
    weights = .95*weights/weights.sum()+.05/len(weights)
    return weights


def coefficients(basis, observed):
    adjoint = basis.conj().swapaxes(-1,-2)
    gram = adjoint @ basis
    size = basis.shape[-1]
    scale = np.trace(gram,axis1=-2,axis2=-1).real.clip(1e-16)/size
    ridge = .003*scale[:,None,None]*np.eye(size)
    rhs = adjoint @ observed[:,None]
    return np.linalg.solve(gram+ridge,rhs)[...,0]


def make_bank(clean, observed, prior):
    scores = lag_scores(clean,observed)
    selected = np.unique(np.r_[np.argsort(scores)[-6:], np.argsort(prior)[-3:]])
    lags = DELAYS[selected]
    basis = clean[reflect(FIT[None,:,None]-lags[:,None,None]-OFFSETS)]
    wanted = observed[FIT]
    models, loss, center = [], [], []
    for size in ORDERS:
        radius = size//2
        columns = slice(6-radius,7+radius)
        design = basis[...,columns]
        cv = np.zeros(len(lags))
        for fold in np.unique(FOLDS):
            valid = FOLDS==fold
            coef = coefficients(design[:,~valid],wanted[~valid])
            estimate = np.einsum('ntk,nk->nt',design[:,valid],coef)
            cv += np.sum(np.abs(estimate-wanted[valid])**2,axis=1)
        loss.extend(cv/len(FIT)*(1+.5*size/len(FIT)))
        fixed = coefficients(design,wanted)
        models.extend(np.pad(fixed,((0,0),(6-radius,6-radius))))
        center.extend(selected)
    return np.asarray(center),np.stack(models),np.asarray(loss)


def pick_model(bank, prior, weight):
    indices, models, residual = bank
    penalty = np.log(prior.max()/prior[indices])
    best = (residual + weight*residual.min()*penalty/len(FIT)).argmin()
    return DELAYS[indices[best]],models[best]


def apply(clean, delay, coefficients, indices):
    offsets = indices[None,:,None]-delay[:,None,None]-OFFSETS
    shifted = clean[np.arange(len(clean))[:,None,None],reflect(offsets)]
    return np.einsum('ntk,nk->nt',shifted,coefficients)


def spectral_noise(clean, observed):
    # Orthogonal sine tapers preserve the received white-noise power. Bins
    # are chosen solely by input excitation. Gamma(3,1/3) median = .891353.
    x = np.fft.fft(clean[:,None,:]*TAPERS,axis=-1,norm='ortho')
    y = np.fft.fft(observed[:,None,:]*TAPERS,axis=-1,norm='ortho')
    quiet = np.argsort(np.mean(np.abs(x)**2,axis=1),axis=-1)[:,:48]
    power = np.take_along_axis(np.mean(np.abs(y)**2,axis=1),quiet,axis=-1)
    return np.median(power,axis=-1)/.891353


def tune(prior_clean, prior_obs, tune_clean, tune_obs):
    prior = learn_delay_prior(prior_clean,prior_obs)
    banks = [make_bank(x,y,prior) for x,y in zip(tune_clean,tune_obs)]
    test = tune_obs[:,TEST]
    energy = np.mean(np.abs(test)**2,axis=-1).clip(1e-18)
    trace = []
    for weight in PRIOR_WEIGHTS:
        lag, coef = map(np.asarray,zip(*[pick_model(bank,prior,weight) for bank in banks]))
        prediction = apply(tune_clean,lag,coef,TEST)
        error = np.mean(np.abs(test-prediction)**2,axis=-1)/energy
        med, p90 = np.quantile(error,[.5,.9])
        trace.append(dict(prior_weight=weight,median=float(med),p90=float(p90),criterion=float(med+p90)))
    selected = min(trace,key=lambda row:row['criterion'])
    return dict(prior=prior.tolist(),selected=selected,trace=trace)


def estimate(clean, observed, config):
    prior = np.asarray(config['prior'])
    weight = config['selected']['prior_weight']
    fitted = [pick_model(make_bank(x,y,prior),prior,weight) for x,y in zip(clean,observed)]
    lag, coef = map(np.asarray,zip(*fitted))
    variance = spectral_noise(clean,observed)
    return lag,coef,variance
