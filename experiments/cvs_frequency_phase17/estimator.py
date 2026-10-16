"""Blind observed-IQ statistic;only scoring receives simulationtruth."""
import numpy as np
from . import design as d
Q=np.array(d.HARMONICS);J=np.array([d.HARMONICS.index(k) for k in d.CENTERS]);B=np.exp(2j*np.pi*np.arange(80)[:,None]*Q[None,:]/d.PERIOD)
def observe(x):
 x=np.asarray(x)
 if x.ndim!=2 or x.shape[1]!=256 or not np.iscomplexobj(x) or not np.isfinite(x).all():raise ValueError('Finite complex IQ Nx256 required')
 u=x[:,80:160].astype(np.complex128);p=np.mean(np.abs(u)**2,axis=1)
 if np.any(p<=0):raise ValueError('Nonzero reference power required')
 lag=np.sum(u[:,20:]*u[:,:-20].conj(),axis=1);omega=np.angle(lag)/20
 z=u*np.exp(-1j*omega[:,None]*np.arange(80)[None,:]);a=z@B.conj()/80
 residual=z-a@B.T;noise=np.maximum(np.sum(np.abs(residual)**2,axis=1)/68,1e-12*np.mean(np.abs(z)**2,axis=1))
 ratios=np.abs(a)**2/(noise[:,None]/80);valid=np.all(ratios>=25.,axis=1)
 h=a[:,J-1]*a[:,J+1]*a[:,J].conj()**2
 closure=h/np.maximum(np.abs(h),1e-12*np.mean(np.abs(z)**2,axis=1)[:,None]**2)
 return closure,valid,dict(CFO_estimate_Hz=omega*d.FS/(2*np.pi),min_carrier_ratio=ratios.min(axis=1),projection_noise_power=noise)
def summarize(mu):
 mu=np.asarray(mu,dtype=np.complex128)
 if mu.shape!=(6,3,4,8) or not np.isfinite(mu).all():raise ValueError('Full6x3x4x8 finite complex cellmeans required')
 if np.any(np.abs(mu)<.8):return dict(identifiable=False,reason='cell phase resultant below fixed .8')
 unit=mu/np.abs(mu);ref=unit.mean(2,keepdims=True)
 if np.any(np.abs(ref)<.8):return dict(identifiable=False,reason='RX reference resultant below fixed .8')
 values=np.angle(unit*ref.conj());values-=values.mean(2,keepdims=True)
 profile=values.mean((0,1));sse=zero=0.;wrongs=np.zeros(3);perrx=np.zeros((2,4));folds=[]
 for tx in range(6):
  for day in range(3):
   fit=values[np.arange(6)!=tx][:,np.arange(3)!=day].mean((0,1));actual=values[tx,day]
   e=np.sum((fit-actual)**2);z=np.sum(actual**2);sse+=e;zero+=z;perrx[0]+=np.sum((fit-actual)**2,axis=1);perrx[1]+=np.sum(actual**2,axis=1)
   w=[float(np.sum((np.roll(fit,k,axis=0)-actual)**2)) for k in (1,2,3)];wrongs+=w
   folds.append(dict(TX=tx,day=day,prediction_SSE=float(e),zero_SSE=float(z),wrong_RX_SSE=w))
 gain=1-sse/zero if zero>1e-18 else None;wrong_gain=(1-wrongs/zero).tolist() if zero>1e-18 else [None]*3
 better=sum(f['prediction_SSE']<f['zero_SSE'] for f in folds);each=bool(np.all(perrx[0]<=perrx[1]+1e-15));rms=float(np.sqrt(np.mean(profile**2)))
 margin=gain-max(wrong_gain) if gain is not None else None
 detected=bool(rms>=.02 and gain is not None and gain>=.5 and better>=15 and each and margin>=.05)
 return dict(identifiable=True,profile=profile.tolist(),cell_RX_contrasts=values.tolist(),profile_RMS_rad=rms,prediction_SSE=float(sse),zero_SSE=float(zero),crossfit_risk_improvement=gain,wrong_RX_risk_improvements=wrong_gain,correct_minus_best_wrong=margin,better_folds=better,per_RX_prediction_SSE=perrx[0].tolist(),per_RX_zero_SSE=perrx[1].tolist(),each_RX_not_worse=each,detected_RX=detected,folds=folds)
def estimate(x):
 if len(x)!=5040:raise ValueError('All5040 syntheticfactor packets required')
 c,valid,quality=observe(x);counts=valid.reshape(6,3,4,70).sum(-1)
 if not bool(valid.all()):return dict(identifiable=False,reason='Requiredcarrier notidentified;no packet orcell dropped',valid_packets=int(valid.sum()),cell_valid_counts=counts.tolist(),cellmeans_real=None,cellmeans_imag=None),quality
 mu=c.reshape(6,3,4,70,8).mean(3);result=summarize(mu);result.update(valid_packets=5040,cell_valid_counts=counts.tolist(),cellmeans_real=mu.real.tolist(),cellmeans_imag=mu.imag.tolist())
 return result,quality
