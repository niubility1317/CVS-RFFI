"""Mathematical and protocol checks only; no actual IQ or model imports."""
import json
import numpy as np
from .noise import *

def run():
 r,a=covariance();assert abs(np.trace(r)-N)<1e-12
 assert np.max(abs(a@a.T-r))<1e-11 and np.max(abs(np.diag(a@a.T)-1))<1e-12
 assert np.max(abs(a-a.T))<1e-12
 x=np.random.default_rng(20261017822).normal(size=(8,2,N)).astype(np.float32);saved=x.copy();ids=['synthetic-'+str(i) for i in range(8)]
 global_state=np.random.get_state();w,c,info=paired(x,ids,0)
 assert np.array_equal(x,saved) and not np.array_equal(w,c)
 check_state=np.random.get_state();assert all(np.array_equal(u,v) for u,v in zip(global_state,check_state))
 w2,c2,info2=paired(x[::-1],ids[::-1],0)
 assert np.array_equal(w2[::-1],w) and np.array_equal(c2[::-1],c) and info2['nominal_SNR_dB'][::-1]==info['nominal_SNR_dB']
 w3,c3,_=paired(x,ids,1);assert not np.array_equal(w,w3) and not np.array_equal(c,c3)
 for ww,cc in ((w,c),(w3,c3)):assert np.isfinite(ww).all() and np.isfinite(cc).all()
 # The exact expected finite-window DFT power follows F R F*,including leakage.
 f=np.fft.fft(np.eye(N),axis=0)/np.sqrt(N);diag=np.real(np.diag(f@r@f.conj().T));mask=abs(np.fft.fftfreq(N,1/FS))<=BAND_EDGE
 out=float(diag[~mask].sum()/N);assert out<.02
 bad=[(x[0],ids,0),(x[:,:,:255],ids,0),(x.astype(complex),ids,0),(np.full_like(x,np.nan),ids,0),(np.zeros_like(x),ids,0),(x,ids[:-1],0),(x,['same']*8,0),(x,['']*8,0),(x,ids,-1),(x,ids,2)]
 rejected=0
 for xx,ii,dd in bad:
  try:paired(xx,ii,dd)
  except ValueError:rejected+=1
  else:raise AssertionError('Invalid input accepted')
 return dict(status='PASS',protocol_negatives=rejected,synthetic_pairs=16,expected_total_noise_power_equal=True,max_covariance_error=float(np.max(abs(a@a.T-r))),expected_out_of_band_fraction=out,white_expected_out_of_band_fraction=float(np.mean(~mask)),actual_IQ=0,CP=0,NN=0,training_updates=0)
if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
