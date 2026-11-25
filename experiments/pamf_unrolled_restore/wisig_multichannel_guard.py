"""Calibrate when wider support and colored noise are identifiable from paired IQ."""

import numpy as np

from . import wisig_multichannel_estimator as solver
from . import wisig_paired_estimator as compact


ACTION_GRID = (0., .5, 1., 1.5, 2.)
COLOR_GRID = (.2, .3, .45, 1.)
BLEND_GRID = (0., .25, .5, 1.)


def compact_candidate(clean,obs,old):
    prior,weight = old['prior'],old['weight']
    residual = np.empty(len(compact.FIT),complex)
    for fold in np.unique(compact.FOLDS):
        held = compact.FOLDS==fold
        delay,taps = compact.pick_model(compact.make_bank(clean,obs,prior,~held),prior,weight)
        times = compact.FIT[held]
        prediction = compact.apply(clean[None,:],np.asarray([delay]),taps[None,:],times)[0]
        residual[held] = obs[times]-prediction
    delay,taps = compact.pick_model(compact.make_bank(clean,obs,prior),prior,weight)
    return solver.embed(delay,taps),0.,residual


def packet(clean,obs,wide,old):
    bank = solver.pack(clean,obs,wide['prior'])
    wide_index,wider = solver.choose(bank,wide['penalty'])
    fallback = compact_candidate(clean,obs,old)
    # Improvement must repeat across IQ windows that were excluded from the fits.
    base = np.abs(fallback[2].reshape(-1,8))**2
    broad = np.abs(wider[2].reshape(-1,8))**2
    delta = np.mean(base-broad,axis=1)
    uncertainty = np.std(delta,ddof=1)/np.sqrt(len(delta))
    evidence = np.mean(delta)/max(float(uncertainty),float(base.mean())*.02,1e-18)
    raw = solver.quiet_spectrum(clean,obs)
    return dict(clean=clean,obs=obs,wide_index=wide_index,wider=wider,fallback=fallback,
                evidence=evidence,raw=raw)


def action_choice(row,margin):
    if row['evidence'] >= margin:
        return row['wide_index'],row['wider']
    return -1,row['fallback']


def exact_rho(blocks,lag):
    a,b = blocks[:,:-lag],blocks[:,lag:]
    power = ((np.abs(a)**2+np.abs(b)**2)*.5).sum()
    return float(np.sum((a.conj()*b).real)/max(power,1e-18))


def noise_features(row,model):
    coeff,omega,residual = model
    windows = residual.reshape(-1,8)
    even,odd = (exact_rho(windows[i::2],1) for i in (0,1))
    rho = np.clip(exact_rho(windows,1),-.85,.85)
    ar_fit = (float(min(abs(even),abs(odd))) if even*odd > 0 and
              abs(even-odd) <= .35 and abs(exact_rho(windows,2)-rho*rho) <= .25 else 0.)
    clean,obs = row['clean'],row['obs']
    error = obs-solver.action(clean,coeff,omega,np.arange(solver.N))
    _,raw = row['raw']
    bins,residual_spectrum = solver.quiet_spectrum(clean,error)
    return bins,raw,residual_spectrum,rho,ar_fit


def variance(features,threshold,blend):
    bins,raw,residual,rho,ar_fit = features
    rho = rho if ar_fit >= threshold else 0.
    shape = (1-rho*rho)/(1+rho*rho-2*rho*np.cos(2*np.pi*bins/solver.N))
    power = (1-blend)*raw+blend*residual
    return max(float(np.median(power/shape)/.891353),1e-18)


def prepare(wide,old):
    return dict(prior=np.asarray(wide['prior']),penalty=wide['penalty'],color=wide['color'],burst=wide['burst']), dict(
        prior=np.asarray(old['prior']),weight=old['selected']['prior_weight'])


def holdout_error(row,model):
    coef,omega,_ = model
    clean,obs = row['clean'],row['obs'][solver.TEST]
    error = np.mean(np.abs(obs-solver.action(clean,coef,omega,solver.TEST))**2)
    return float(error),float(np.mean(np.abs(obs)**2).clip(1e-18))


def tune(clean,received,wide,old):
    wide,old = prepare(wide,old)
    midpoint = clean.shape[1]//2
    groups = [[packet(x,y,wide,old) for x,y in zip(group_x[midpoint:],group_y[midpoint:])]
              for group_x,group_y in zip(clean,received)]
    action_trace = []
    for margin in ACTION_GRID:
        grouped = []
        for rows in groups:
            values = [holdout_error(row,action_choice(row,margin)[1]) for row in rows]
            error,energy = np.asarray(values).T
            grouped.append(float(np.quantile(error/energy,[.5,.9]).sum()))
        action_trace.append(dict(margin=margin,criterion=float(np.mean(grouped))))
    margin = min(action_trace,key=lambda row:row['criterion'])['margin']
    residuals = []
    for rows in groups:
        fits = [(row,action_choice(row,margin)[1]) for row in rows]
        residuals.append([(noise_features(row,fit),holdout_error(row,fit)[0]) for row,fit in fits])
    noise_trace = []
    for threshold in COLOR_GRID:
        for blend in BLEND_GRID:
            scores = []
            for rows in residuals:
                improvements = []
                for features,mse in rows:
                    v0 = variance(features,1.,0.)
                    v = variance(features,threshold,blend)
                    improvements.append(mse/v-mse/v0+np.log(v/v0))
                scores.append(float(np.median(improvements)))
            noise_trace.append(dict(color_threshold=threshold,spectrum_blend=blend,
                                    criterion=float(np.mean(scores))))
    preferred = min(noise_trace,key=lambda row:row['criterion'])
    return dict(action_margin=margin,color_threshold=preferred['color_threshold'],
                spectrum_blend=preferred['spectrum_blend'],action_trace=action_trace,noise_trace=noise_trace)
