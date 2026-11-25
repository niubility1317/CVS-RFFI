"""IQ-only evidence averaging for ambiguous low-SNR paired WiSig receive actions."""

import numpy as np

from . import wisig_ten_channel_estimator as adaptive


SCALES = np.array([0.,.004,.012,.035,.10,.28,.75,2.,6.,18.])
BLENDS = (.0,.5,.85,1.)
GATES = (.35,.8,2.,5.)


def delay_weights(clean,observed):
    support = []
    for x,y in zip(clean,observed):
        bank = x[adaptive.OFFSETS]
        energies = np.sum(np.abs(bank)**2,axis=0).clip(1e-18)
        powers = np.abs(bank.conj().T@y)**2/energies
        normalized = powers/max(float(np.median(powers)),1e-18)
        keep = np.argsort(normalized)[-6:]
        score = np.zeros(len(adaptive.LAGS))
        score[keep] = np.maximum(normalized[keep]-3.,0.)
        support.append(score/max(score.sum(),1.))
    prior = np.sum(support,axis=0)
    separation = adaptive.LAGS[:,None]-adaptive.LAGS
    prior = np.exp(-.5*(separation/1.25)**2)@prior
    prior = prior/max(prior.sum(),1e-18)
    prior = .82*prior+.18/len(prior)
    return prior/prior.sum()


def posterior(clean,obs,train,variance,prior):
    design = clean[adaptive.OFFSETS]*np.sqrt(prior)
    received = obs[train]
    matrix = design[train]
    gram = matrix.conj().T@matrix
    eigen,basis = np.linalg.eigh(gram)
    keep = eigen>max(float(eigen.max())*1e-6,1e-18)
    if not keep.any():
        return np.zeros(len(clean),complex),float(np.mean(np.abs(received)**2)),0.
    power = eigen[keep]
    rotation = basis[:,keep]
    left = matrix@rotation/np.sqrt(power)
    coordinates = left.conj().T@received
    noise = max(float(variance),1e-18)
    eigen = power/(power.sum()/len(train))
    snr = SCALES[:,None]*eigen[None,:]
    gamma = 1./(1.+snr)
    log_probability = -(np.log1p(snr)+(np.abs(coordinates)**2/noise)[None,:]*gamma).sum(1)
    log_probability -= .75*np.log1p(SCALES)
    weight = np.exp(log_probability-log_probability.max())
    weight /= weight.sum()
    shrink = weight@(1.-gamma)
    coef = rotation@(shrink*coordinates/np.sqrt(power))
    predicted = design@coef
    residual = received-predicted[train]
    dof = np.sum(2.*shrink-shrink**2)
    # Near-zero shrink reduces to estimating power directly from the input noise.
    corrected = np.sum(np.abs(residual)**2)/max(len(train)-dof,1.)
    visible_power = float(np.mean(np.abs(predicted[train])**2))
    return predicted,float(max(corrected,1e-18)),visible_power/noise


def pair(clean,obs,train,current,prior):
    bank,rho,tones = adaptive.candidate_bank(clean,obs,train)
    settings = (current['complexity'],current['shrink'])
    signal,var,dof = adaptive.estimate(bank,rho,tones,clean,obs,train,*settings,current['blend'])
    # In both calibration and deployment the posterior sees only the fit residual noise scale.
    _,fit_variance,_ = adaptive.estimate(bank,rho,tones,clean,obs,train,*settings,0.)
    predicted,variance,confidence = posterior(clean,obs,train,fit_variance,prior)
    return signal,var,predicted,variance,confidence,dof


def select(row,maximum,mix,variance_mix):
    signal,var,alternative,var2,confidence,_ = row
    weight = mix*np.clip(1.+.5*(1-confidence/maximum),0.,1.)
    amount = variance_mix*weight
    return signal+weight*(alternative-signal),var+amount*(var2-var)


def tune(clean,observed,current):
    split = len(clean)//2
    prior = delay_weights(clean[:split],observed[:split])
    train = adaptive.T[(adaptive.T//8)%4!=1]
    held = adaptive.T[(adaptive.T//8)%4==1]
    validation = [(y,pair(x,y,train,current,prior)) for x,y in zip(clean[split:],observed[split:])]
    action = []
    for gate in GATES:
        for mix in BLENDS:
            stats = []
            for y,row in validation:
                proposed,_ = select(row,gate,mix,0.)
                loss = np.mean(np.abs(y[held]-proposed[held])**2)
                scale = np.mean(np.abs(y[held]-row[0][held])**2).clip(1e-18)
                stats.append(loss/scale)
            action.append(dict(gate=gate,mix=mix,score=float(np.quantile(stats,[.5,.9]).mean())))
    best = min(action,key=lambda item:item['score'])
    variance = []
    for component in (0.,.5,1.):
        change = []
        for y,row in validation:
            proposed,value = select(row,best['gate'],best['mix'],component)
            error = np.mean(np.abs(y[held]-proposed[held])**2)
            change.append(np.log(value/row[1])+error/value-error/row[1])
        variance.append(dict(mix=component,score=float(np.median(change))))
    weight = min(variance,key=lambda item:item['score'])
    return dict(prior=prior.tolist(),mix=best['mix'],gate=best['gate'],variance_mix=weight['mix'],
                action_grid=action,variance_grid=variance)
