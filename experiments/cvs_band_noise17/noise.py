"""Same-power white versus stationary band-limited proper Gaussian observation noise.
Only synthetic noise is colored. No filtering, whitening, clipping, padding,
centering or renormalization of received IQ is performed.
"""
from functools import lru_cache
import hashlib
import json
import numpy as np
FS=25000000.
BAND_EDGE=26*20000000./64
N=256
SNR=(20.,36.)
SEED=20261017821
DRAWS=(0,1)

def seed(*parts):
 payload=json.dumps([SEED,*parts],ensure_ascii=False,separators=(',',':')).encode('utf8')
 return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')

@lru_cache(maxsize=1)
def covariance():
 k=np.arange(N);r=np.sinc(2*BAND_EDGE/FS*(k[:,None]-k[None,:]))
 values,vectors=np.linalg.eigh(r)
 if values.min() < -1e-11:raise FloatingPointError('Band covariance is not PSD')
 root=(vectors*np.sqrt(np.maximum(values,0.)))@vectors.T
 numerical=root@root.T
 if np.max(abs(numerical-r))>1e-11 or abs(np.trace(numerical)-N)>1e-10:raise FloatingPointError('Same expected power not preserved')
 root.setflags(write=False);r.setflags(write=False)
 return r,root

def paired(x,physical_ids,draw):
 x=np.asarray(x)
 if x.dtype.kind!='f' or x.ndim!=3 or x.shape[1:]!=(2,N) or not len(x) or not np.isfinite(x).all():raise ValueError('Finite real [B,2,256] IQ required')
 if len(physical_ids)!=len(x) or len(set(physical_ids))!=len(x) or any(not isinstance(p,str) or not p for p in physical_ids) or draw not in DRAWS:raise ValueError('Unique opaque source physical IDs and fixed draw required')
 p=np.mean(x[:,0].astype(float)**2+x[:,1].astype(float)**2,axis=1)
 if np.any(p<=0):raise ValueError('Nonzero received power required')
 r,root=covariance();noise=[];snr=[]
 for pid in physical_ids:
  rng=np.random.default_rng(seed('noise',draw,pid))
  z=rng.normal(size=(N,2));noise.append((z[:,0]+1j*z[:,1])/np.sqrt(2))
  snr.append(np.random.default_rng(seed('SNR',draw,pid)).uniform(*SNR))
 white=np.asarray(noise);color=white@root.T
 sigma=np.sqrt(p)*10**(-np.asarray(snr)/20)
 white*=sigma[:,None];color*=sigma[:,None]
 raw=x[:,0].astype(float)+1j*x[:,1].astype(float)
 def as_iq(z):return np.stack((z.real,z.imag),axis=1).astype(np.float32)
 return as_iq(raw+white),as_iq(raw+color),dict(nominal_SNR_dB=list(map(float,snr)),expected_noise_power=(sigma*sigma).tolist(),white_noise_power=np.mean(abs(white)**2,axis=1).tolist(),color_noise_power=np.mean(abs(color)**2,axis=1).tolist(),white_relative_IQ_change=np.sqrt(np.mean(abs(white)**2,axis=1)/p).tolist(),color_relative_IQ_change=np.sqrt(np.mean(abs(color)**2,axis=1)/p).tolist(),input_filtered=False,received_IQ_RMS_renormalized=False,Gaussian_tails_clipped_or_redrawn=False)

def spectral_support(x):
 x=np.asarray(x,dtype=float);z=x[:,0]+1j*x[:,1];power=abs(np.fft.fft(z,axis=1))**2
 mask=abs(np.fft.fftfreq(N,1/FS))<=BAND_EDGE
 return (power[:,mask].sum(1)/power.sum(1)).tolist()
