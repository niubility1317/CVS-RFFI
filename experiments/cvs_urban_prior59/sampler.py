"""Private categorical coupling preserves the actual legacy equal-choice result."""
import math
import numpy as np
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
NOMINAL=(1/3,1/3,1/3);URBAN=(.2,.4,.4)

def weights(values):
 if len(values)!=3 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not math.isfinite(float(v)) or v<=0 for v in values) or not math.isclose(sum(values),1.,rel_tol=0,abs_tol=1e-12):raise ValueError('Three finite positive scene weights summing to one required')
 return tuple(float(v) for v in values)

def bounded_mix(eta):
 if isinstance(eta,bool) or not isinstance(eta,(int,float)) or not math.isfinite(eta) or not 0<=eta<=1:raise ValueError('Fixed mixture strength in [0,1] required')
 return weights(tuple((1-eta)*a+eta*b for a,b in zip(NOMINAL,URBAN)))

def inverse_cdf(u,prior):
 prior=weights(prior)
 if not math.isfinite(float(u)) or not 0<=u<1:raise ValueError('Scene uniform must be in [0,1)')
 return min(int(np.searchsorted(np.cumsum(prior),u,side='right')),2)

def coupled_uniform(rng):
 # choice without p uses a bounded-integer draw, not Generator.random().
 # A fresh private bucket offset yields a common continuous uniform; the first
 # categorical bucket is exactly the legacy equal scene selection.
 legacy=int(rng.choice(3));offset=float(rng.random());lo=legacy/3.;hi=(legacy+1)/3.
 u=min(max((legacy+offset)/3.,lo),float(np.nextafter(hi,lo)))
 return u,legacy

def choose(rng,prior=URBAN):
 prior=weights(prior);u,legacy=coupled_uniform(rng)
 if inverse_cdf(u,NOMINAL)!=legacy:raise ValueError('Categorical equal-prior coupling lost parity')
 index=inverse_cdf(u,prior)
 return SCENES[index],dict(scene_uniform=u,legacy_equal_scene=SCENES[legacy],scene_index=index)
