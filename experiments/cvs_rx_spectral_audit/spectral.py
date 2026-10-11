"""Observable contrasts, not blind receiver/channel identification."""
import math
import torch

def observations(x,window):
 if x.ndim!=3 or tuple(x.shape[1:])!=(2,256) or not torch.isfinite(x).all():raise ValueError('Invalid IQ')
 z=torch.complex(x[:,0].double(),x[:,1].double())
 if window=='hann':z=z*torch.hann_window(256,periodic=False,dtype=torch.float64,device=x.device)
 elif window!='rectangular':raise ValueError('Unknown window')
 psd=torch.fft.fftshift(torch.fft.fft(z).abs().square(),dim=-1).reshape(-1,32,8).mean(-1)
 power=psd.mean(-1,keepdim=True)
 if (power<=0).any():raise ValueError('Zero-energy IQ')
 logs=(psd/power).clamp_min(1e-6).log();logs-=logs.mean(-1,keepdim=True)
 n=torch.arange(32,dtype=torch.float64,device=x.device)+.5;k=torch.arange(1,9,dtype=torch.float64,device=x.device)
 basis=torch.cos(math.pi*n[:,None]*k[None,:]/32)*math.sqrt(2/32)
 return {'centered_log_PSD32':logs,'DCT1to8':logs@basis}

def grid(values,y,rx,day,cell_count=70):
 ys=y.unique(sorted=True);rs=rx.unique(sorted=True);ds=day.unique(sorted=True)
 if (len(ys),len(ds),len(rs))!=(6,3,5):raise ValueError('Wrong source strata')
 means=torch.empty(6,3,5,values.shape[-1],dtype=torch.float64);halves=torch.empty(2,*means.shape,dtype=torch.float64)
 for i,yi in enumerate(ys):
  for j,di in enumerate(ds):
   for k,ri in enumerate(rs):
    samples=values[(y==yi)&(day==di)&(rx==ri)]
    if len(samples)!=cell_count:raise ValueError('Incomplete balanced L cell')
    means[i,j,k]=samples.mean(0)
    halves[0,i,j,k]=samples[::2].mean(0);halves[1,i,j,k]=samples[1::2].mean(0)
 return means,halves,dict(TX=ys.tolist(),day=ds.tolist(),RX=rs.tolist(),cell_count=cell_count)

def fit_contrast(means,tx,day):
 # Exclusion is a conjunction: neither held-out TX nor held-out day enters fit.
 mask_t=torch.arange(means.shape[0])!=tx;mask_d=torch.arange(means.shape[1])!=day
 effect=means[mask_t][:,mask_d].mean((0,1))
 return effect-effect.mean(0,keepdim=True)

def audit(means,halves):
 if tuple(means.shape[:3])!=(6,3,5) or not torch.isfinite(means).all():raise ValueError('Invalid mean grid')
 folds=[];actuals=[];predictions=[];controls={k:0. for k in ['prediction','zero','reverse',*[f'RX_shift_{i}' for i in range(1,5)]]}
 for t in range(6):
  for d in range(3):
   pred=fit_contrast(means,t,d);actual=means[t,d]-means[t,d].mean(0,keepdim=True)
   errors=dict(prediction=float((pred-actual).square().sum()),zero=float(actual.square().sum()),reverse=float((-pred-actual).square().sum()))
   errors.update({f'RX_shift_{i}':float((pred.roll(i,0)-actual).square().sum()) for i in range(1,5)})
   for name,v in errors.items():controls[name]+=v
   cosine=float((pred*actual).sum()/(pred.norm()*actual.norm()).clamp_min(1e-20))
   folds.append(dict(TX_index=t,day_index=d,fit_cells=5*2*5,test_cells=5,SSE=errors,relative_MSE=errors['prediction']/errors['zero'] if errors['zero']>1e-20 else None,cosine=cosine,predicted_norm=float(pred.norm()),observed_norm=float(actual.norm())))
   actuals.append(actual);predictions.append(pred)
 actual=torch.stack(actuals);pred=torch.stack(predictions)
 h=halves-halves.mean(3,keepdim=True)
 # Difference / 2 estimates the full-cell mean noise under exchangeable independent halves.
 # Report as a descriptive proxy, not a calibrated CI: packets may be correlated.
 noise=float((h[0]-h[1]).square().sum()/4)
 energy=controls['zero'];coordinate=[]
 for j in range(means.shape[-1]):
  den=float(actual[:,:,j].square().sum());num=float((pred[:,:,j]-actual[:,:,j]).square().sum())
  coordinate.append(dict(index=j,observed_energy=den,residual_energy=num,relative_MSE=num/den if den>1e-20 else None))
 return dict(folds=folds,total_SSE=controls,relative_MSE={k:v/energy if energy>1e-20 else None for k,v in controls.items()},folds_better_than_zero=sum(f['SSE']['prediction']<f['SSE']['zero'] for f in folds),fold_count=18,TX_summaries=[dict(TX_index=t,SSE=sum(f['SSE']['prediction'] for f in folds if f['TX_index']==t),zero_SSE=sum(f['SSE']['zero'] for f in folds if f['TX_index']==t)) for t in range(6)],coordinate_diagnostics=coordinate,split_replicate_noise_proxy=noise,noise_proxy_fraction=noise/energy if energy>1e-20 else None,replicate_limitation='Alternating row halves within each L cell; no timestamp assumption, independence claim or confidence interval',predicted_contrasts=pred.tolist(),observed_contrasts=actual.tolist())
