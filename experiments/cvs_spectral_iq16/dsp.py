"""Centered complex-linear finite FIR;power restoration is explicit."""
import math,torch
from torch.nn import functional as F
from . import design as d
from .spectral import basis,observe
def response(h,n):
 p=torch.zeros(n,dtype=torch.complex128);p[torch.arange(-16,17)%n]=h;return torch.fft.fft(p)
def kernel(target):
 h=torch.fft.ifft(torch.fft.ifftshift(target));h=h[torch.arange(-16,17)%d.GRID]*torch.hann_window(33,periodic=False,dtype=torch.float64);return h
def bank(effects):
 e=torch.as_tensor(effects,dtype=torch.float64)
 if e.shape!=(4,8) or not bool(torch.isfinite(e).all()):raise ValueError('Finite fit-only RX effects required')
 result={};delta=torch.zeros(33,dtype=torch.complex128);delta[16]=1
 for a,b in d.PAIRS:
  coeff=e[d.FIT.index(b)]-e[d.FIT.index(a)];other=coeff*torch.tensor(d.CONTROL,dtype=torch.float64)
  log32=[v@basis().T for v in (coeff,other)];peak=max(float(v.abs().max()) for v in log32);cap=min(1.,d.DB*math.log(10)/10/max(peak,1e-30))
  for sign in d.SIGNS:
   targets=[torch.exp(sign*.5*cap*(v@basis(d.GRID).T)).to(torch.complex128) for v in (coeff,other)]
   perturb=[kernel(t)-delta for t in targets];ns,nc=(p.norm() for p in perturb)
   match=float(ns/nc) if float(nc)>1e-20 else 1.
   if (float(ns)>1e-20)!=(float(nc)>1e-20):raise ValueError('Degenerate matched control')
   perturb[1]=perturb[1]*match
   maxima=max(float(response(p,d.GRID).abs().max()) for p in perturb);q=min(1.,(1-10**(-d.DB/20))/max(maxima,1e-30))
   for i,kind in enumerate(('structured','control')):
    h=delta+q*perturb[i];H=response(h,d.GRID);target=1+q*(targets[i]-1)*(match if i else 1.);den=float((target-1).abs().square().sum());error=float((torch.fft.fftshift(H)-target).abs().square().sum())
    name=f'{kind}_RX{a}_{b}_{"plus" if sign>0 else "minus"}';gain=20*H.abs().log10()
    result[name]=dict(h=h,response256=response(h,256),desired_response=target,metadata=dict(pair=[a,b],sign=sign,kind=kind,cap_only_scale=cap,common_attenuation=q,control_perturbation_match_scale=match,perturbation_energy=float((h-delta).abs().square().sum()),kernel_relative_RMSE=(error/den)**.5 if den>1e-25 else 0.,grid_response_max_abs_dB=float(gain.abs().max()),grid_response_min_dB=float(gain.min()),grid_response_max_dB=float(gain.max()),grid_phase_max_abs_rad=float(torch.angle(H).abs().max())))
 return result
def apply(raw,item,raw_obs,raw_fft):
 if raw.ndim!=3 or raw.shape[1:]!=(2,256) or raw_obs.shape!=(len(raw),32) or raw_fft.shape!=(len(raw),256) or not bool(torch.isfinite(raw).all()):raise ValueError('Fixed finite raw source IQ required')
 h=item['h'];x=raw.double();p=x.square().sum(1).mean(1)
 if bool((p<=0).any()):raise ValueError('Positive source power required')
 padded=F.pad(x,(16,16),mode='reflect');r=h.real.flip(0)[None,None];im=h.imag.flip(0)[None,None]
 yr=F.conv1d(padded[:,0:1],r)-F.conv1d(padded[:,1:2],im);yi=F.conv1d(padded[:,1:2],r)+F.conv1d(padded[:,0:1],im);finite=torch.cat((yr,yi),1)
 oracle=torch.fft.ifft(raw_fft*item['response256']);oracle=torch.stack((oracle.real,oracle.imag),1)
 residual=finite-oracle;interior=float(residual[:,:,16:240].square().sum()/x[:,:,16:240].square().sum().clamp_min(1e-30));edge_error=float(residual.square().sum());change=float((oracle-x).square().sum())
 after=finite.square().sum(1).mean(1);scale=(p/after.clamp_min(1e-30)).sqrt();y=(finite*scale[:,None,None]).to(raw.dtype)
 finite_delta=observe(y)-raw_obs;oracle_delta=observe(oracle)-raw_obs;num=float((finite_delta-oracle_delta).square().sum());den=float(oracle_delta.square().sum());power_error=float(((y.double().square().sum(1).mean(1)-p).abs()/p).max());g=20*scale.log10()
 values=dict(count=len(raw),spectral_error_SSE=num,spectral_oracle_change_SSE=den,boundary_error_energy=edge_error,oracle_IQ_change_energy=change,interior_relative_MSE=interior,RMS_relative_error_max=power_error,restore_gain_dB_min=float(g.min()),restore_gain_dB_max=float(g.max()),finite_delta_sum=finite_delta.sum(0).tolist(),oracle_delta_sum=oracle_delta.sum(0).tolist())
 if not bool(torch.isfinite(y).all()):raise ValueError('Nonfinite filtered IQ')
 return y,values
