"""Whole-packet bounded nonaffine unit-modulus observation, no labels."""
import torch
from . import design as d

def project_bound(phi):
 if not isinstance(phi,torch.Tensor) or phi.ndim!=2 or phi.shape[1]!=256 or not len(phi) or phi.dtype!=torch.float64 or phi.device.type!='cpu' or phi.requires_grad or not bool(torch.isfinite(phi).all()):raise ValueError('Finite CPU Float64 trajectories B256 required')
 t=torch.arange(256,dtype=torch.float64)-127.5
 residual=phi-phi.mean(1,keepdim=True)-(phi*t).sum(1,keepdim=True)/t.square().sum()*t
 peak=residual.abs().amax(1,keepdim=True);scale=torch.where(peak>d.PHASE_MAX,d.PHASE_MAX/peak.clamp_min(torch.finfo(phi.dtype).tiny),torch.ones_like(peak));theta=residual*scale
 return theta,scale

def apply_cpu(x,phi):
 if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype!=torch.float32 or x.device.type!='cpu' or x.requires_grad or not bool(torch.isfinite(x).all()):raise ValueError('Finite CPU rawFloat32IQB2x256 withoutgrad required')
 px=x.double().square().sum((1,2))
 if not bool((px>0).all()):raise ValueError('Positive rawpacket power required;no filtering')
 if phi.shape[0]!=len(x):raise ValueError('Matching phasepacketcount required')
 theta,scale=project_bound(phi);z=torch.complex(x[:,0].double(),x[:,1].double());q=z*torch.complex(torch.cos(theta),torch.sin(theta));out=torch.stack((q.real,q.imag),1).to(x.dtype)
 n=out.double()-x.double();pn=n.square().sum((1,2));py=out.double().square().sum((1,2));pr=x[:,:,80:160].double().square().sum((1,2));nr=n[:,:,80:160].square().sum((1,2));t=torch.arange(256,dtype=torch.float64)-127.5
 if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(theta).all()):raise ValueError('Finite output andphase required')
 info=dict(phase_RMS_rad=theta.square().mean(1).sqrt().tolist(),phase_peak_rad=theta.abs().amax(1).tolist(),phase_mean_rad=theta.mean(1).tolist(),phase_linear_slope_rad_per_sample=((theta*t).sum(1)/t.square().sum()).tolist(),phase_scale=scale.view(-1).tolist(),raw_RMS=torch.sqrt(px/256).tolist(),output_RMS=torch.sqrt(py/256).tolist(),relative_IQ_RMS_change=torch.sqrt(pn/px).tolist(),relative_power_change=(py/px-1).tolist(),relative_reference_RMS_change=torch.sqrt(nr/pr.clamp_min(torch.finfo(pr.dtype).tiny)).tolist(),reference_power_zero=(pr==0).tolist(),max_point_modulus_error=(q.abs()-z.abs()).abs().amax(1).tolist())
 if max(info['phase_peak_rad'])>d.PHASE_MAX+1e-12 or max(info['relative_IQ_RMS_change'])>.1+1e-6 or max(abs(v) for v in info['relative_power_change'])>1e-6 or max(abs(v) for v in info['phase_mean_rad'])>1e-12 or max(abs(v) for v in info['phase_linear_slope_rad_per_sample'])>1e-12:raise ValueError('Bound,affine orpower invariant failed')
 return out,info


def apply(x,step):
 if not isinstance(step,int) or step not in range(1,4441) or not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype!=torch.float32 or x.requires_grad or x.device.type not in ('cpu','cuda') or not bool(torch.isfinite(x).all()):raise ValueError('Finite rawFloat32IQB2x256 andfixedoptimizerstep required')
 if not bool((x.double().square().sum((1,2))>0).all()):raise ValueError('Positive rawpacket power required,no filtering')
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(d.PHASE_SEED+step)
  phi=torch.randn((len(x),256),device='cpu',dtype=torch.float64).mul_(d.SIGMA).cumsum(1)
  value,info=apply_cpu(x.detach().to('cpu'),phi)
 return value.to(x.device),dict(seed=d.PHASE_SEED+step,**info)
