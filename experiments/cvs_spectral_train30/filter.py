"""Unchanged Hann33 source kernels;one label-free private draw per actual Ag slot."""
import hashlib,json,time
from collections import Counter
import torch
from torch.nn import functional as F
from . import design as d
class FixedFilter:
 def __init__(self,bank=None):
  self._provided=bank;self._keys=None;self._h=None;self._weights={};self.calls=0;self.packets=0;self.counts=Counter();self.by_rx=Counter();self.steps=[];self.enqueue_seconds=0.;self.max_RMS_relative_error=0.
 def __deepcopy__(self,memo):
  v=type(self)(self._provided);memo[id(self)]=v;return v
 def load(self):
  if self._keys is not None:return
  bank=d.read(d.BANK_PATH) if self._provided is None else self._provided
  keys=sorted(k for k in bank if k.startswith('structured_'))
  if len(keys)!=12 or len(set(keys))!=12:raise ValueError('All12 original structured directions required')
  hs=[]
  for key in keys:
   b=bank[key];h=torch.tensor(b['h_real'],dtype=torch.float64)+1j*torch.tensor(b['h_imag'],dtype=torch.float64)
   if h.shape!=(33,) or not bool(torch.isfinite(h).all()) or b['grid_response_max_abs_dB']>.50000001:raise ValueError('Original fixed bounded Hann33 bank required')
   hs.append(h)
  self._keys=keys;self._h=torch.stack(hs)
 def transform(self,x,indices):
  self.load()
  if x.ndim!=3 or x.shape[1:]!=(2,256) or not torch.is_floating_point(x) or not bool(torch.isfinite(x).all()) or len(indices)!=len(x) or any(i not in range(12) for i in indices):raise ValueError('Finite IQ256 and original kernel indices required')
  cache=(str(x.device),x.dtype)
  if cache not in self._weights:
   h=self._h.to(device=x.device);hr=h.real.to(x.dtype).flip(-1);hi=h.imag.to(x.dtype).flip(-1)
   self._weights[cache]=torch.stack((torch.stack((hr,-hi),1),torch.stack((hi,hr),1)),1)
  w=self._weights[cache][torch.tensor(indices,device=x.device)].reshape(2*len(x),2,33)
  padded=F.pad(x,(16,16),mode='reflect').reshape(1,2*len(x),288)
  y=F.conv1d(padded,w,groups=len(x)).reshape(len(x),2,256);p=x.square().sum(1).mean(1);after=y.square().sum(1).mean(1)
  if bool((p<=0).any()) or bool((after<=0).any()):raise ValueError('No zero-power or dropped packet allowed')
  y=y*(p/after).sqrt()[:,None,None]
  if not bool(torch.isfinite(y).all()):raise ValueError('No nonfinite filtered observations allowed')
  err=float(((y.square().sum(1).mean(1)-p).abs()/p).max());self.max_RMS_relative_error=max(self.max_RMS_relative_error,err)
  return y
 @staticmethod
 def choice(pid,epoch,step):
  if len(pid)!=5 or pid[0]!=1 or pid[1] not in d.FIT or pid[2] not in (1,2,3) or epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1:raise ValueError('Original source metadata and cumulativeE21..30 required')
  raw=json.dumps([d.FILTER_SEED,'FIR_Ag',epoch,step,list(map(int,pid))],separators=(',',':'));return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8],'little')%12
 def apply(self,x,meta,epoch,step):
  start=time.perf_counter();keys=('eq_i','rx_i','day_i','sig_i','base_index')
  if len(x)!=64 or any(k not in meta or len(meta[k])!=64 for k in keys):raise ValueError('Exactly64 declared Ag slots with original physical metadata required')
  vals={k:meta[k].detach().cpu().tolist() if torch.is_tensor(meta[k]) else list(meta[k]) for k in keys};ids=[tuple(int(vals[k][i]) for k in keys) for i in range(64)]
  choices=[self.choice(pid,epoch,step) for pid in ids];y=self.transform(x,choices);cc=Counter(choices)
  for pid,k in zip(ids,choices):self.counts[self._keys[k]]+=1;self.by_rx[(pid[1],pid[2],self._keys[k])]+=1
  self.calls+=1;self.packets+=64;self.steps.append(dict(epoch=epoch,step=step,kernel_counts={self._keys[k]:n for k,n in cc.items()}));self.enqueue_seconds+=time.perf_counter()-start
  return y
 def audit(self):
  self.load();return dict(operator='Fixed33tap sourcebank complex-linear FIR',bank_run=d.BANK_RUN,kernels=self._keys,label_free_application=True,source_bank_fitted_only_on_legal_L_subset=True,previous_feasibility_failed_unchanged=True,no_exact_RX_replacement_claim=True,normalization='restore inputRMS',boundary='reflect16/centered/no roll',no_new_CFO_noise_PA_DAC=True)
 def summary(self):return dict(calls=self.calls,packets=self.packets,kernel_counts=dict(self.counts),per_RX_day_kernel=[dict(RX=k[0],day=k[1],kernel=k[2],count=n) for k,n in sorted(self.by_rx.items())],steps=self.steps,max_RMS_relative_error=self.max_RMS_relative_error,host_enqueue_seconds=self.enqueue_seconds,full_GPU_wall_not_inferred_from_enqueue=True,real_FIR_MACC=self.packets*2*256*2*33,extra_NN=0,source_RX_holdout=None,**self.audit())
