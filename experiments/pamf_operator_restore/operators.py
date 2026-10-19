"""Finite-window widely-linear operators; oracle objects are diagnostic only."""
from functools import lru_cache
import math
import numpy as np
from leo_practical.equalization import linear_filter
from leo_practical.residual import ResidualChannel


def real_pair(z):
    z=np.asarray(z)
    return np.concatenate((z.real,z.imag))


def complex_pair(z):
    z=np.asarray(z);n=len(z)//2
    if z.ndim!=1 or len(z)!=2*n:raise ValueError('Even one-dimensional IQ vector required')
    return z[:n]+1j*z[n:]


def real_operator(main,image):
    main,image=np.asarray(main),np.asarray(image)
    if main.ndim!=2 or main.shape!=image.shape:raise ValueError('Matching complex operators required')
    return np.block([[(main+image).real,-(main-image).imag],
                     [(main+image).imag,(main-image).real]])


def filter_matrix(kernel,count,boundary='reflect'):
    basis=np.eye(count,dtype=np.complex128)
    return np.column_stack([linear_filter(basis[:,j],kernel,boundary)[0] for j in range(count)])


@lru_cache(maxsize=128)
def delay_matrix(count,delay):
    if delay<0:raise ValueError('Causal nonnegative delay required')
    kernel=np.zeros(delay+1,complex);kernel[-1]=1
    matrix=filter_matrix(kernel,count).real;matrix.setflags(write=False)
    return matrix


def oracle_snapshot(clean,cfg,seed,receiver):
    """Freeze the actual input-dependent AGC/lock, not a global linear model.

    Two clean-independent probes recover modulation of the same latent FIR.
    The original native implementation is untouched. Neither the true action,
    quality nor noise covariance is available to an observation-only method.
    """
    if cfg.processing_route!='residual' or cfg.mode!='post_sync' or cfg.equalization_enabled or cfg.boundary!='reflect':
        raise ValueError('This feasibility model is residual/post_sync/noEQ/reflect only')
    clean=np.asarray(clean,np.complex128);count=len(clean)
    observed,meta,traces=ResidualChannel(cfg,seed,receiver).process(clean,return_traces=True)
    h=filter_matrix(traces['residual_kernel'],count,cfg.boundary)
    rng=np.random.default_rng(2026101907)
    probe=(rng.normal(size=count)+1j*rng.normal(size=count))/math.sqrt(2)
    signals=[]
    for value in (probe,1j*probe):
        _,_,trace=ResidualChannel(cfg,seed,receiver).process(value,return_traces=True,quality_snr_db=meta['quality_snr_db'])
        signals.append(trace['pre_agc_signal'])
    response=h@probe
    if np.min(abs(response))<1e-12:raise FloatingPointError('Oracle probe cannot identify a zero response row')
    a=(signals[0]-1j*signals[1])/(2*response)
    b=(signals[0]+1j*signals[1])/(2*response.conj())
    gain=10**(meta['agc_gain_db']/20)
    matrix=gain*real_operator(a[:,None]*h,b[:,None]*h.conj())
    # Proper reference noise passes through the same widely-linear RX terms.
    transform=np.empty((count,2,2))
    transform[:,0,0]=(a+b).real;transform[:,0,1]=-(a-b).imag
    transform[:,1,0]=(a+b).imag;transform[:,1,1]=(a-b).real
    covariance=(gain**2*10**(-cfg.reference_snr_db/10)/2)*(transform@transform.transpose(0,2,1))
    expected=gain*traces['pre_agc_signal'];replayed=complex_pair(matrix@real_pair(clean))
    relative=float(np.linalg.norm(replayed-expected)/max(np.linalg.norm(expected),1e-12))
    if relative>1e-9:raise FloatingPointError('Exact oracle signal decomposition differs')
    noise=complex_pair(real_pair(observed)-matrix@real_pair(clean))
    native_noise=gain*traces['pre_agc_noise']
    observed_error=float(np.linalg.norm(noise-native_noise)/max(np.linalg.norm(observed),1e-12))
    if observed_error>2e-7:raise FloatingPointError('Native complex64 observation decomposition differs')
    return observed,matrix,covariance,dict(meta,oracle_signal_relative_error=relative,
        observation_decomposition_relative_error=observed_error,
        oracle_uses_true_AGC_lock_channel=True,estimated_methods_receive_oracle_fields=False)


class GaussianSystem:
    """Soft observation update; no hard projection and no true clean argument."""
    def __init__(self,matrix,observed,covariance):
        self.matrix=np.asarray(matrix,np.float64);self.observed=np.asarray(observed,np.float64)
        rows,columns=self.matrix.shape
        if rows!=columns or self.observed.shape!=(rows,) or rows%2:raise ValueError('Square real IQ system required')
        if not np.isfinite(self.matrix).all() or not np.isfinite(self.observed).all():raise ValueError('Finite observation system required')
        if np.isscalar(covariance):
            if not math.isfinite(covariance) or covariance<=0:raise ValueError('Positive noise variance required')
            self.whitened=self.matrix/math.sqrt(covariance);self.value=self.observed/math.sqrt(covariance)
        else:
            covariance=np.asarray(covariance,np.float64);count=rows//2
            if covariance.shape!=(count,2,2) or not np.isfinite(covariance).all():raise ValueError('Per-time real noise covariance required')
            values,vectors=np.linalg.eigh(covariance)
            if np.min(values)<=0:raise ValueError('Strictly positive noise covariance required')
            inverse=(vectors*(1/np.sqrt(values))[:,None,:])@vectors.transpose(0,2,1)
            top=inverse[:,0,0,None]*self.matrix[:count]+inverse[:,0,1,None]*self.matrix[count:]
            bottom=inverse[:,1,0,None]*self.matrix[:count]+inverse[:,1,1,None]*self.matrix[count:]
            self.whitened=np.concatenate((top,bottom))
            self.value=np.concatenate((inverse[:,0,0]*self.observed[:count]+inverse[:,0,1]*self.observed[count:],
                                       inverse[:,1,0]*self.observed[:count]+inverse[:,1,1]*self.observed[count:]))
        gram=self.whitened.T@self.whitened
        values,self.vectors=np.linalg.eigh(gram);self.values=np.maximum(values,0)

    def solve(self,prior,precision):
        prior=np.asarray(prior,np.float64)
        if prior.shape!=self.observed.shape or not np.isfinite(prior).all():raise ValueError('Finite matched prior required')
        if not math.isfinite(precision) or precision<=0:raise ValueError('Positive fixed prior precision required')
        residual=self.value-self.whitened@prior
        update=self.vectors@((self.vectors.T@(self.whitened.T@residual))/(self.values+precision))
        return prior+update

    def geometry(self,precision):
        return dict(effective_observed_dimensions=float(np.sum(self.values/(self.values+precision))),
                    real_dimensions=len(self.values),gram_eigenvalue_min=float(self.values.min()),
                    gram_eigenvalue_max=float(self.values.max()),
                    relative_gram_eigenvalues_below_1e_8=int(np.sum(self.values<self.values.max()*1e-8)))


def estimate_action(received,coarse,delays):
    """Transient LS from current received IQ and its frozen coarse estimate.

    No channel, clean reference, label, scene or persistent adaptation input.
    A static FIR is explicitly an approximation to the native varying action.
    """
    received,coarse=np.asarray(received,complex),np.asarray(coarse,complex)
    if received.ndim!=1 or coarse.shape!=received.shape or not np.isfinite(received).all() or not np.isfinite(coarse).all():
        raise ValueError('Current finite received/coarse record required')
    delays=tuple(delays)
    if not delays or not {0,24}<=set(delays) or len(set(delays))!=len(delays) or min(delays)<0:raise ValueError('Fixed causal taps must include identity0 and numerical delay24')
    matrices=[delay_matrix(len(received),d) for d in delays]
    main=np.column_stack([m@coarse for m in matrices]);image=np.column_stack([m@coarse.conj() for m in matrices])
    design=np.concatenate((main,image),axis=1);gram=design.conj().T@design
    regularization=max(1e-10,.01*float(np.trace(gram).real)/len(gram))
    anchor=np.zeros(len(gram),complex);anchor[delays.index(0)]=1
    coefficients=np.linalg.solve(gram+regularization*np.eye(len(gram)),design.conj().T@received+regularization*anchor)
    count=len(delays)
    m=sum(c*matrix for c,matrix in zip(coefficients[:count],matrices))
    b=sum(c*matrix for c,matrix in zip(coefficients[count:],matrices))
    estimated=real_operator(m,b)
    error=received-design@coefficients
    variance=max(float(np.mean(abs(error)**2)),1e-4*float(np.mean(abs(received)**2)),1e-12)/2
    return estimated,variance,dict(taps=list(delays),complex_parameters=len(coefficients),
        fitting_regularization=regularization,estimated_real_noise_variance=variance,
        received_fit_nmse=float(np.mean(abs(error)**2)/max(np.mean(abs(received)**2),1e-12)),
        true_channel_clean_TX_inputs=False,persistent_state_updates=0)
