"""Recover a low-rank time-varying receive action from one paired IQ packet."""

from dataclasses import dataclass

import numpy as np

from . import wisig_multichannel_estimator as earlier
from .wisig_independent_channel_audit import N, reflect


LAGS = np.arange(-50,51)
T = np.arange(N)
U = np.linspace(-1.,1.,N)
PHASE = np.column_stack((U,U**2-.34))
OFFSETS = reflect(T[:,None]-LAGS)
DERIVATIVE_OFFSETS = np.r_[np.arange(-7,0),np.arange(1,8)]
DERIVATIVE = (-1.)**(DERIVATIVE_OFFSETS+1)/DERIVATIVE_OFFSETS*np.sinc(DERIVATIVE_OFFSETS/8)
GRID = ((1.8,0.),(2.7,0.),(4.,0.),(1.8,.65),(2.7,.65),(4.,.65))
VARIANCE_BLEND = (0.,.2,.45)


@dataclass
class Candidate:
    design: np.ndarray
    coefficient: np.ndarray
    uncertainty: np.ndarray
    prediction: np.ndarray
    weighted_error: float
    dof: float
    penalty: float
    coordinates: tuple | None = None
    rotation: np.ndarray | None = None


def randomized_tones(residual,train):
    """Identify up to two narrow peaks for a joint nuisance regression."""
    masked = np.zeros(N,complex)
    masked[train] = residual*np.hanning(N)[train]
    power = np.abs(np.fft.fft(masked,4*N))**2
    noise = float(np.median(power)/np.log(2.))
    result = []
    for _ in range(2):
        i = int(np.argmax(power))
        if power[i]<15.*max(noise,1e-18):
            break
        side = np.log(power[(i+np.arange(-1,2))%len(power)].clip(1e-18))
        offset = .5*(side[0]-side[2])/min(side[0]+side[2]-2*side[1],-1e-12)
        result.append(np.exp(2j*np.pi*(i+np.clip(offset,-.5,.5))/len(power)*T))
        power[(i+np.arange(-13,14))%len(power)] = 0.
    return np.column_stack(result) if result else np.empty((N,0),complex)


def ar_correlation(residual,train):
    ends = train[1:][np.diff(train)==1]
    filled = np.zeros(N,complex)
    filled[train] = residual

    def rho(lag,select):
        a,b = filled[select-lag],filled[select]
        power = np.sum((np.abs(a)**2+np.abs(b)**2)*.5)
        return float(np.real(np.vdot(a,b))/max(power,1e-18))

    rho0,rho1 = (rho(1,ends[(ends//16)%2==i]) for i in (0,1))
    second = ends[np.isin(ends-1,ends)]
    strength = rho(1,ends)
    if rho0*rho1 > 0 and min(abs(rho0),abs(rho1))>.17 and abs(rho(2,second)-strength**2)<.20:
        return float(np.clip(strength,-.78,.78))
    return 0.


def whitener(train,rho):
    consecutive = np.r_[False,np.diff(train)==1]

    def apply(values):
        result = values[train].copy()
        result[consecutive] -= rho*values[train[consecutive]-1]
        result[~consecutive] *= np.sqrt(1-rho*rho)
        return result

    return apply


def least_squares(matrix,y,ridge=.0008):
    scale = np.linalg.norm(matrix,axis=0).clip(1e-12)
    scaled = matrix/scale
    adjoint = scaled.conj().T
    gram = adjoint@scaled
    stabilized = gram+ridge*np.eye(len(scale))
    inverse = np.linalg.solve(stabilized,np.eye(len(scale)))
    fit = inverse@(adjoint@y)
    hat = inverse@gram
    covariance = hat@inverse
    dof = 2*np.trace(hat).real-np.trace(hat@hat).real
    return fit/scale,np.diag(covariance).real/scale**2,float(dof)


def fit_candidate(design,nuisance,obs,whiten,search=0.):
    joint = np.column_stack((design,nuisance))
    matrix,y = whiten(joint),whiten(obs)
    coeff,uncertainty,dof = least_squares(matrix,y)
    error = np.sum(np.abs(y-matrix@coeff)**2)
    count = design.shape[1]
    return Candidate(design,coeff[:count],uncertainty[:count],design@coeff[:count],
                     float(error),dof,dof+search)


def compact(dictionary,y,train,*,with_support=False):
    matrix = dictionary[train]
    projection = np.abs(matrix.conj().T@y[train])**2/np.sum(np.abs(matrix)**2,axis=0).clip(1e-18)
    centers = np.unique(np.r_[np.argsort(projection)[-5:],len(LAGS)//2])
    design = []
    for width in (1,3,7,13):
        neighbors = [np.arange(max(0,i-width//2),min(len(LAGS),i+width//2+1)) for i in centers]

        def loss(columns):
            local = matrix[:,columns]
            coef,_,_ = least_squares(local,y[train])
            return np.sum(np.abs(y[train]-local@coef)**2)

        selected = min(neighbors,key=loss)
        block = dictionary[:,selected]
        design.append((block, selected) if with_support else block)
    return design


def pursuit(matrix,y,counts):
    norms = np.linalg.norm(matrix,axis=0).clip(1e-12)
    adjoint = matrix.conj().T
    residual = y.copy()
    columns = []
    for k in range(1,max(counts)+1):
        score = np.abs(adjoint@residual)/norms
        score[columns] = -np.inf
        columns.append(int(score.argmax()))
        fitted = matrix[:,columns]
        coef,_,_ = least_squares(fitted,y)
        residual = y-fitted@coef
        if k in counts:
            yield np.asarray(columns)


def initial_phase(prediction,y,train):
    blocks = train//16
    products = y[train]*prediction[train].conj()
    pooled = np.bincount(blocks,weights=products.real,minlength=16)+1j*np.bincount(
        blocks,weights=products.imag,minlength=16)
    weight = np.sqrt(np.abs(pooled)).clip(1e-12)
    center = 16*np.arange(16)+7
    terms = np.column_stack((np.ones(16),PHASE[center]))
    slope = np.linalg.lstsq(terms*weight[:,None],np.unwrap(np.angle(pooled))*weight,rcond=None)[0][1:]
    return np.clip(slope,[-6.,-3.5],[6.,3.5])


def polish(design,seed,nuisance,obs,whiten,limit,search):
    phase = seed.copy()
    for _ in range(2):
        rotated = design*np.exp(1j*(PHASE@phase))[:,None]
        joint = np.column_stack((rotated,nuisance))
        matrix = whiten(joint)
        y = whiten(obs)
        coef,_,_ = least_squares(matrix,y)
        predicted = rotated@coef[:design.shape[1]]
        jacobian = whiten(1j*predicted[:,None]*PHASE[:,:limit])
        # Remove the linear action subspace before updating the phase.
        jacobian -= matrix@np.linalg.lstsq(matrix,jacobian,rcond=.001)[0]
        rhs = y-matrix@coef
        step = np.linalg.lstsq(np.r_[jacobian.real,jacobian.imag],np.r_[rhs.real,rhs.imag],rcond=.01)[0]
        phase[:limit] = np.clip(phase[:limit]+np.clip(step,-.8,.8),-6.,6.)
    rotation = np.exp(1j*(PHASE@phase))
    rotated = design*rotation[:,None]
    fit = fit_candidate(rotated,nuisance,obs,whiten,search+limit)
    fit.rotation = rotation
    return fit


def candidate_bank(clean,obs,train,*,keep_coordinates=False):
    dictionary = clean[OFFSETS]
    derivative_times = T[:,None,None]-LAGS[:,None]+DERIVATIVE_OFFSETS
    derivative = clean[reflect(derivative_times)]@DERIVATIVE
    base = compact(dictionary,obs,train,with_support=True)
    empty = np.empty((N,0),complex)
    direct = whitener(train,0.)
    starting = [(fit_candidate(design,empty,obs,direct,1.), support) for design,support in base]
    initial, initial_support = min(starting,key=lambda row:row[0].weighted_error*(1+2*row[0].penalty/len(train)))
    seed = initial_phase(initial.prediction,obs,train)
    coarse = polish(initial.design,np.array([seed[0],0.]),empty,obs,direct,1,1.)
    residual = obs[train]-min((initial,coarse),key=lambda fit:fit.weighted_error).prediction[train]
    nuisance = randomized_tones(residual,train)
    if nuisance.shape[1]:
        estimates,_,_ = least_squares(nuisance[train],residual)
        residual -= nuisance[train]@estimates
    rho = ar_correlation(residual,train)
    whiten = whitener(train,rho)
    bank = []

    def add(fit, support, kinds=('constant',)):
        if keep_coordinates:
            fit.coordinates = tuple((int(column), kind) for kind in kinds for column in support)
        bank.append(fit)

    for design,support in base:
        add(fit_candidate(design,nuisance,obs,whiten,1.), support)
    add(polish(initial.design,np.array([seed[0],0.]),nuisance,obs,whiten,1,1.), initial_support)
    rotated = np.exp(1j*(PHASE@seed))
    expanded = np.column_stack((dictionary,dictionary.conj()))
    expanded_derivative = np.column_stack((derivative,derivative.conj()))
    # Score fixed sparse paths with and without a smooth phase, then test slow tap changes.
    for expansion,phase,counts in ((dictionary,np.zeros(2),(2,4,7)),(expanded,seed,(2,4,7,10))):
        matrix = whiten(expansion*rotated[:,None] if expansion is expanded else expansion)
        y = whiten(obs)
        if nuisance.shape[1]:
            q,_ = np.linalg.qr(whiten(nuisance))
            matrix = matrix-q@(q.conj().T@matrix)
            y = y-q@(q.conj().T@y)
        for support in pursuit(matrix,y,counts):
            design = expansion[:,support]
            search = .7*len(support)
            if expansion is dictionary:
                add(fit_candidate(design,nuisance,obs,whiten,search), support)
                continue
            add(polish(design,np.array([phase[0],0.]),nuisance,obs,whiten,1,search), support)
            add(polish(design,phase,nuisance,obs,whiten,2,search), support)
            if len(support) in (4,7):
                dynamic = np.column_stack((design,design*U[:,None]))
                add(polish(dynamic,phase,nuisance,obs,whiten,2,search), support, ('constant','linear'))
                rate = expanded_derivative[:,support]
                stretch = np.column_stack((design,rate,rate*U[:,None]))
                add(polish(stretch,phase,nuisance,obs,whiten,2,search), support,
                    ('constant','derivative','linear_derivative'))
    return bank,rho,nuisance.shape[1]


def operator_coefficients(fit, coefficients):
    """Expand selected Adaptive columns into real-linear, time-by-lag coordinates."""
    if fit.coordinates is None or len(fit.coordinates) != len(coefficients):
        raise ValueError('Adaptive operator coordinates are incomplete')
    extent = int(LAGS[-1] + DERIVATIVE_OFFSETS[-1])
    lags = np.arange(-extent, extent+1)
    direct = np.zeros((N, len(lags)), complex)
    image = np.zeros_like(direct)
    rotation = fit.rotation if fit.rotation is not None else np.ones(N)
    for (index, kind), coefficient in zip(fit.coordinates, coefficients):
        destination = image if index >= len(LAGS) else direct
        lag = LAGS[index % len(LAGS)]
        gain = coefficient*rotation*U if kind.startswith('linear') else coefficient*rotation
        if kind.endswith('derivative'):
            destination[:, lag-DERIVATIVE_OFFSETS+extent] += gain[:, None]*DERIVATIVE
        else:
            destination[:, lag+extent] += gain
    return lags, direct, image


def estimate(bank,rho,tones,clean,obs,train,complexity,shrink,blend,*,return_operator=False):
    n = len(train)
    fit = min(bank,key=lambda v:v.weighted_error*(1+complexity*v.penalty/(n-v.dof)))
    innovation = fit.weighted_error/max(n-fit.dof-1,1.)
    dispersion = innovation/max(1-rho*rho,.1)
    shrinkage = np.abs(fit.coefficient)**2/(np.abs(fit.coefficient)**2+shrink*innovation*fit.uncertainty+1e-20)
    coefficients = fit.coefficient*shrinkage
    signal = fit.design@coefficients
    if tones:
        raw_residual = obs[train]-fit.prediction[train]
        dispersion = max(dispersion,float(np.mean(np.abs(raw_residual)**2))+dispersion*fit.dof/n)
    bins,power = earlier.quiet_spectrum(clean,obs)
    quiet = float(np.median(power)/.891353)
    x = np.fft.fft(clean[None,:]*earlier.compact.TAPERS,axis=-1,norm='ortho')
    spectrum = np.mean(np.abs(x)**2,axis=0)
    quiet_occupancy = np.median(spectrum[bins])/max(float(spectrum.mean()),1e-18)
    # Spectral floor support is weaker when color or coherent interference is visible.
    weight = blend*np.clip(1.-quiet_occupancy/.12,0.,1.) if not (rho or tones) else 0.
    variance = max((1-weight)*dispersion+weight*quiet,1e-18)
    if return_operator:
        return signal,variance,operator_coefficients(fit, coefficients)
    return signal,variance,float(fit.dof)


def choose_and_apply(clean,obs,options,train=T,*,return_operator=False):
    bank,rho,tones = candidate_bank(clean,obs,train,keep_coordinates=return_operator)
    return estimate(bank,rho,tones,clean,obs,train,options['complexity'],options['shrink'],options['blend'],
                    return_operator=return_operator)


def tune(clean,observed):
    fit = T[(T//8)%4!=1]
    held = T[(T//8)%4==1]
    prepared = [[(x,y,*candidate_bank(x,y,fit)) for x,y in zip(group_x,group_y)]
                for group_x,group_y in zip(clean,observed)]
    trace = []
    for complexity,shrink in GRID:
        for blend in VARIANCE_BLEND:
            action_loss,noise_loss = [],[]
            for group in prepared:
                losses = []
                for x,y,bank,rho,tones in group:
                    signal,var,_ = estimate(bank,rho,tones,x,y,fit,complexity,shrink,blend)
                    error = np.mean(np.abs(y[held]-signal[held])**2)
                    scaled = error/np.mean(np.abs(y[held])**2).clip(1e-18)
                    losses.append((scaled, np.log(var/max(error,1e-18))+error/var-1.))
                action,noise = np.asarray(losses).T
                action_loss.append(float(np.log(np.quantile(action,[.5,.9])).mean()))
                noise_loss.append(float(np.median(noise)))
            trace.append(dict(complexity=complexity,shrink=shrink,blend=blend,
                              action=float(np.mean(action_loss)),noise=float(np.mean(noise_loss))))
    action = min(trace,key=lambda row:row['action'])
    preferred = min((row for row in trace if (row['complexity'],row['shrink']) == (action['complexity'],action['shrink'])),
                    key=lambda row:row['noise'])
    return dict(preferred,trace=trace)
