"""Reorder received cycles; never infer labels, replace TX hardware or change power."""
import hashlib,random
import torch
START,STOP,PERIOD=80,160,20
SEED=202610173001
def apply_parameters(x,permutation,phase):
 if x.ndim!=3 or len(x)<1 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('Finite original float32 IQ256 required')
 n=len(x);p=torch.as_tensor(permutation,device=x.device,dtype=torch.long);phase=torch.as_tensor(phase,device=x.device,dtype=x.dtype)
 if p.shape!=(n,4) or phase.shape!=(n,) or not torch.isfinite(phase).all() or not torch.equal(p.sort(1).values,torch.arange(4,device=x.device).expand(n,4)):raise ValueError('Complete finite four-cycle permutation required')
 z=torch.complex(x[:,0],x[:,1]);cycles=z[:,START:STOP].reshape(n,4,PERIOD);index=p[:,:,None].expand(n,4,PERIOD);selected=cycles.gather(1,index)
 angles=phase[:,None]*(torch.arange(4,device=x.device)[None,:]-p);factor=torch.polar(torch.ones_like(angles),angles);new=(selected*factor[:,:,None]).reshape(n,80)
 out=x.clone();out[:,0,START:STOP]=new.real;out[:,1,START:STOP]=new.imag
 return out

def transform(x,ids,view):
 if not isinstance(view,str) or not view or len(ids)!=len(x) or len(set(ids))!=len(ids) or any(not isinstance(pid,str) or not pid for pid in ids):raise ValueError('Unique opaque physical IDs and named source view required')
 if x.ndim!=3 or len(x)<1 or x.shape[1:]!=(2,256) or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError('Finite IQ256 required')
 z=torch.complex(x[:,0],x[:,1]);left=z[:,80:140];right=z[:,100:160];dot=(right*left.conj()).sum(1);den=(left.abs().square().sum(1)*right.abs().square().sum(1)).sqrt();valid=(den>1e-12)&(dot.abs()>1e-12)
 phase=torch.where(valid,torch.atan2(dot.imag,dot.real),torch.zeros_like(den));coherence=torch.where(den>1e-12,dot.abs()/den.clamp_min(1e-12),torch.zeros_like(den));perms=[]
 for i,pid in enumerate(ids):
  key=(str(SEED)+'/'+view+'/'+pid).encode('utf8');seed=int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little');p=list(range(4));random.Random(seed).shuffle(p)
  if not bool(valid[i]):p=list(range(4)) # Retain undefined packets as recorded identity, never delete.
  perms.append(p)
 out=apply_parameters(x,perms,phase);old_power=x.square().sum((1,2));new_power=out.square().sum((1,2));relative=(out-x).square().sum((1,2)).sqrt()/old_power.sqrt().clamp_min(1e-12)
 boundary=torch.tensor([80,100,120,140,160],device=x.device);before=(z[:,boundary]-z[:,boundary-1]).abs().square().mean(1);zz=torch.complex(out[:,0],out[:,1]);after=(zz[:,boundary]-zz[:,boundary-1]).abs().square().mean(1)
 meta=dict(permutation=perms,phase_progress_rad=phase.detach().cpu().tolist(),coherence=coherence.detach().cpu().tolist(),phase_defined=valid.detach().cpu().tolist(),relative_L2=relative.detach().cpu().tolist(),power_relative_error=((new_power-old_power).abs()/old_power.clamp_min(1e-12)).detach().cpu().tolist(),join_energy_before=before.detach().cpu().tolist(),join_energy_after=after.detach().cpu().tolist())
 if not torch.equal(out[:,:,:80],x[:,:,:80]) or not torch.equal(out[:,:,160:],x[:,:,160:]) or not torch.isfinite(out).all() or max(meta['power_relative_error'],default=0)>2e-6:raise ValueError('Support/power/finite invariant failed')
 return out,meta
