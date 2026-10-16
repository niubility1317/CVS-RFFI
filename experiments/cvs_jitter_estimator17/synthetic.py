"""Ground truth used only by generator/scorer,never estimator arguments."""
import numpy as np
from . import design as d
from .estimator import outer

def template(name):
 n=np.arange(d.PERIOD);k=np.r_[np.arange(-6,0),np.arange(1,7)];i=np.arange(12)
 if name in ('multitone_quadratic','multitone_cubic'):
  phase=i*i*.47 if name=='multitone_quadratic' else i*i*i*.13+i*.23
  w=np.exp(1j*phase);z=np.sum(w[:,None]*np.exp(2j*np.pi*k[:,None]*n/d.PERIOD),axis=0);dz=np.sum(w[:,None]*(2j*np.pi*k[:,None]/d.PERIOD)*np.exp(2j*np.pi*k[:,None]*n/d.PERIOD),axis=0)
 elif name=='tone':z=np.exp(2j*np.pi*3*n/d.PERIOD);dz=2j*np.pi*3/d.PERIOD*z
 elif name=='constant':z=np.ones(d.PERIOD,dtype=np.complex128);dz=np.zeros_like(z)
 elif name=='zero':z=np.zeros(d.PERIOD,dtype=np.complex128);dz=np.zeros_like(z)
 else:raise ValueError('Unregistered template')
 power=float(np.mean(np.abs(z)**2));scale=np.sqrt(power) if power else 1.
 return z/scale,dz/scale

def generate(name,SNR,case,seed):
 z,dz=template(name);rng=np.random.default_rng(seed);rho=10**(-SNR/10);_,shares,CFO=case
 derivative_power=float(np.mean(np.abs(dz)**2));power=float(np.mean(np.abs(z)**2))
 weights=np.array([rho*shares[0],rho*shares[1],rho*shares[2],rho*shares[3]/derivative_power if derivative_power else 0.])
 rotation=np.exp(1j*rng.uniform(-np.pi,np.pi,(d.PACKETS,1,1)))
 gain=rng.normal(size=(d.PACKETS,d.CYCLES,1));phase=rng.normal(size=gain.shape);jitter=rng.normal(size=(d.PACKETS,d.CYCLES,d.PERIOD));noise=(rng.normal(size=jitter.shape)+1j*rng.normal(size=jitter.shape))/np.sqrt(2.)
 x=(z[None,None,:]+np.sqrt(weights[1])*gain*z[None,None,:]+1j*np.sqrt(weights[2])*phase*z[None,None,:]+np.sqrt(weights[3])*jitter*dz[None,None,:]+np.sqrt(weights[0])*noise)*rotation
 time=np.arange(d.CYCLES*d.PERIOD).reshape(d.CYCLES,d.PERIOD);carrier=np.exp(2j*np.pi*CFO*time)
 x=x*carrier[None,:,:]
 true_basis=np.stack((np.tile([.5,0.,.5],(d.PERIOD,1)),outer(z),outer(1j*z),outer(dz)),axis=-1).reshape(-1,4)
 true_Gram=true_basis.T@true_basis
 # All6 unordered cycle pairs determine the expected pair-difference energy.
 means=z[None,:]*carrier;drift=sum(float(np.mean(np.abs(means[a]-means[b])**2))/2 for a in range(4) for b in range(a+1,4))/6
 total=weights[0]+(weights[1]+weights[2])*power+weights[3]*derivative_power+drift
 metadata=dict(true_variances=weights.tolist(),true_Gram=true_Gram.tolist(),signal_power=power,derivative_power=derivative_power,CFO_derivative_power=float(np.mean(np.abs(dz+2j*np.pi*CFO*z)**2)),total_innovation_energy=float(total),CFO_drift_energy=drift)
 return x,metadata

def score(estimated,truth):
 if estimated['variances'] is None:return dict(jitter_energy_fraction=None,jitter_relative_error=None,covariance_relative_error=None)
 w=np.array(estimated['variances']);true=np.array(truth['true_variances']);G=np.array(truth['true_Gram']);den=truth['total_innovation_energy'];jp=truth['CFO_derivative_power']
 fraction=float(w[3]*jp/den) if den>0 else None
 relative=float(abs(w[3]-true[3])/true[3]) if true[3]>0 else None
 covden=float(true@G@true);delta=w-true;cov=float(np.sqrt(max(0.,float(delta@G@delta))/covden)) if covden>0 else None
 return dict(jitter_energy_fraction=fraction,jitter_relative_error=relative,covariance_relative_error=cov)
