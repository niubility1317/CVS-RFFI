"""Deterministic source statistics, not a receiver classifier or hardware estimator."""
from collections import Counter
import numpy as np
from . import design as d

def periodograms(x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or not np.isfinite(x).all():raise ValueError('Finite IQ Nx2x256 required')
 z=x[:,0]+1j*x[:,1];answer=[]
 for window in (np.ones(256),np.hanning(256)):
  p=np.abs(np.fft.fftshift(np.fft.fft(z*window,axis=-1),axes=-1))**2;power=p.mean(-1,keepdims=True)
  if np.any(power<=1e-30):raise ValueError('Undefined periodogram;no sample deletion')
  answer.append(p/power)
 return np.stack(answer)

def cell_spectra(p,tx,rx,day):
 p=np.asarray(p,dtype=np.float64);tx,rx,day=map(np.asarray,(tx,rx,day))
 if p.shape!=(2,5040,256) or any(a.shape!=(5040,) for a in (tx,rx,day)) or not np.isfinite(p).all() or np.any(p<0):raise ValueError('All5040 source observations required')
 if set(tx.tolist())!=set(d.TX) or set(rx.tolist())!=set(d.FIT) or set(day.tolist())!=set(d.DAYS):raise ValueError('Exact source cells only;no heldRX/U/V')
 out=np.empty((2,6,3,4,256))
 for t in d.TX:
  for j,dd in enumerate(d.DAYS):
   for k,r in enumerate(d.FIT):
    mask=(tx==t)&(rx==r)&(day==dd)
    if mask.sum()!=70:raise ValueError('Exactly70 packets per registered cell')
    out[:,t,j,k]=p[:,mask].mean(1)
 return out

def prominence(psd):
 a=np.asarray(psd,dtype=np.float64)
 if a.shape!=(2,6,3,4,256) or not np.isfinite(a).all() or np.any(a<0) or np.any(a.sum(-1)<=1e-30):raise ValueError('Finite complete cell spectra required')
 bins=np.asarray(d.BINS);line=np.mean(np.stack([a[...,bins+i] for i in (-1,0,1)]),axis=0)
 background=np.median(np.stack([a[...,bins+i] for i in (-7,-6,-5,-4,4,5,6,7)]),axis=0)
 # PSD is unit mean by packet; fixed relative floor is not an adaptive exclusion.
 return 10*np.log10(np.maximum(line,1e-12)/np.maximum(background,1e-12))

def fit(values,tx,day):
 if values.shape!=(2,6,3,4,len(d.BINS)) or not 0<=tx<6 or not 0<=day<3 or not np.isfinite(values).all():raise ValueError('Complete observation grid required')
 return values[:,np.arange(6)!=tx][:,:,np.arange(3)!=day].mean((1,2))

def choose(fitted):
 if fitted.shape!=(2,4,len(d.BINS)) or not np.isfinite(fitted).all():raise ValueError('Fit only selector required')
 contrast=fitted[0]-fitted[0].mean(0,keepdims=True)
 return tuple(int(i) for i in np.unravel_index(np.argmax(contrast),contrast.shape))

def analyze(psd):
 a=prominence(psd);folds=[];errors=np.zeros((2,18,4));zero=np.zeros_like(errors);wrong=np.zeros((2,18,3));success=0;selections=[]
 for t in d.TX:
  for day in range(3):
   ix=3*t+day;fitted=fit(a,t,day);r,b=choose(fitted);selections.append((r,int(d.BINS[b])))
   pred=fitted[...,b]-fitted[...,b].mean(-1,keepdims=True);actual=a[:,t,day,:,b]-a[:,t,day,:,b].mean(-1,keepdims=True)
   errors[:,ix]=(pred-actual)**2;zero[:,ix]=actual**2
   for k in range(1,4):wrong[:,ix,k-1]=((np.roll(pred,k,axis=-1)-actual)**2).sum(-1)
   # Both window checks use one rectangular-fit selection, never held maxima.
   ok=bool(np.all(fitted[:,r,b]>=6.) and np.all(pred[:,r]>=3.) and np.all(a[:,t,day,r,b]>=3.) and np.all(actual[:,r]>=3.));success+=ok
   folds.append(dict(TX_index=t,day=d.DAYS[day],selected_RX=d.FIT[r],selected_shifted_bin=int(d.BINS[b]),selected_frequency_hz=(d.BINS[b]-128)*d.FS/256,fit_prominence_db=fitted[:,r,b].tolist(),fit_RX_contrast_db=pred[:,r].tolist(),held_prominence_db=a[:,t,day,r,b].tolist(),held_RX_contrast_db=actual[:,r].tolist(),line_success=ok,SSE=errors[:,ix].sum(-1).tolist(),zero_SSE=zero[:,ix].sum(-1).tolist(),wrong_RX_SSE=wrong[:,ix].tolist()))
 threshold=d.config(d.rows()[0])['criteria'];windows=[]
 def reduction(e,z):return None if z<=1e-20 else float(1-e/z)
 for w,name in enumerate(('rectangular','hann')):
  e,z=errors[w],zero[w];den=float(z.sum());gain=reduction(float(e.sum()),den);wrong_ratio=None if den<=1e-20 else float((wrong[w].sum(0).min()-e.sum())/den)
  txg=[reduction(float(e[3*t:3*t+3].sum()),float(z[3*t:3*t+3].sum())) for t in range(6)]
  dayg=[reduction(float(e[dd::3].sum()),float(z[dd::3].sum())) for dd in range(3)]
  rxg=[reduction(float(e[:,rr].sum()),float(z[:,rr].sum())) for rr in range(4)]
  windows.append(dict(window=name,risk_reduction=gain,positive_folds=int(np.sum(e.sum(-1)<z.sum(-1))),correct_over_best_wrong_RX=wrong_ratio,TX_risk_reductions=txg,day_risk_reductions=dayg,RX_risk_reductions=rxg,SSE=float(e.sum()),zero_SSE=den,wrong_RX_SSE=wrong[w].sum(0).tolist()))
 mode,count=Counter(selections).most_common(1)[0]
 checks=dict(complete_finite_grid=True,successful_folds=success>=threshold['successful_folds_min'],same_RX_bin=count>=threshold['modal_exact_RX_bin_folds_min'],risk=all(v['risk_reduction'] is not None and v['risk_reduction']>=.5 for v in windows),positive_folds=all(v['positive_folds']>=15 for v in windows),wrong_RX_controls=all(v['correct_over_best_wrong_RX'] is not None and v['correct_over_best_wrong_RX']>=.2 for v in windows),all_TX_day_RX=all(x is not None and x>0 for v in windows for key in ('TX_risk_reductions','day_risk_reductions','RX_risk_reductions') for x in v[key]))
 return dict(status='PASS_SOURCE_ASSOCIATION_ONLY' if all(checks.values()) else 'FAIL',checks=checks,successful_folds=success,modal_RX=d.FIT[mode[0]],modal_shifted_bin=mode[1],modal_frequency_hz=(mode[1]-128)*d.FS/256,modal_count=count,windows=windows,folds=folds,model_forwards=0,checkpoint_loads=0,optimizer_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False),a
