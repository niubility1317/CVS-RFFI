"""Exact original complete practical physics; only legal cumulative window extends."""
from collections import Counter
from dataclasses import asdict
import math
import numpy as np
import torch
from experiments.cvs_clock_factorial30 import design as d

def serial(v):
 if isinstance(v,np.generic):return serial(v.item())
 if isinstance(v,np.ndarray):return serial(v.tolist())
 if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if isinstance(v,complex):return [v.real,v.imag]
 if isinstance(v,float) and not math.isfinite(v):return None
 return v

class Practical:
 def __init__(self,role='train'):
  if role!='train':raise ValueError('This adapter generates only labeled training views')
  from experiments.cvs_phase1_overlay.model import native_modules
  native_modules()
  import leo_practical.channel as ch
  import leo_practical.residual as residual
  self.ch=ch;self.residual=residual;self.calls=0;self.packets=0;self.coverage=Counter();self.sessions={}
  self.configs={s:ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}
 def __deepcopy__(self,memo):
  v=type(self)();memo[id(self)]=v;return v
 def audit(self):return dict(channel_module_file=self.ch.__file__,residual_module_file=self.residual.__file__,configurations={s:asdict(c) for s,c in self.configs.items()},all_original_RX=list(d.FIT),ordinary_before_satellite=False,formal_test_changed=False)
 def apply(self,x,meta,epoch,step):
  if epoch not in range(31,51) or step not in range(6661,11101) or epoch!=(step-1)//222+1 or x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite real L training IQ/declared cumulative E31..50 only')
  keys=('eq_i','rx_i','day_i','sig_i','base_index')
  if any(k not in meta or len(meta[k])!=len(x) for k in keys):raise ValueError('Physical source metadata required')
  v={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in keys}
  if set(map(int,v['rx_i']))-set(d.FIT) or set(map(int,v['day_i']))-{1,2,3} or set(map(int,v['eq_i']))!={1}:raise ValueError('Original full source RX/day/equalized roles required')
  raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);out=[];records=[];namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S';self.last_namespace=namespace
  for i in range(len(x)):
   pid=tuple(int(v[k][i]) for k in keys);rx=int(v['rx_i'][i]);day=int(v['day_i'][i]);session=f'rx{rx}:day{day}:eq{int(v["eq_i"][i])}'
   rng=np.random.default_rng(self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,'scene',pid));scene=str(rng.choice(d.TRAIN_SCENES));cfg=self.configs[scene]
   seed=self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,scene,pid);hw=self.ch.receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('Hardware changed within label-independent physical session')
   normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);signal,m=self.residual.ResidualChannel(cfg,seed,hw).process(normalized)
   if not np.isfinite(signal).all() or m['channel_equalization_applied']:raise ValueError('Original full residual/noEQ/finite output required;no dropping')
   self.coverage[(rx,day,scene,m['state_end'],m['receiver_lock_status'])]+=1;out.append(np.stack((signal.real,signal.imag)));records.append(serial(dict(m,scene=scene,sample_id=pid,session_id=session,namespace=namespace)))
  self.calls+=1;self.packets+=len(x);a=np.ascontiguousarray(out,dtype=np.float32)
  return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device),records
 def summary(self):return dict(calls=self.calls,packets=self.packets,session_count=len(self.sessions),sessions=self.sessions,strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(self.coverage.items())],all_original_source_RX=list(d.FIT),dropped=0,confidence_filter=False,ordinary_before_satellite=False,formal_test_changed=False)

from experiments.cvs_crossview_supcon41.channel import tensor_iq

from .protocol import cumulative
class Shifted:
 def __init__(self,obj):self.obj=obj
 def __getattr__(self,k):
  if k=='obj':raise AttributeError(k)
  return getattr(self.obj,k)
 def apply(self,x,meta,epoch,step):
  epoch,step=cumulative(epoch,step)
  if len(x)!=64:raise ValueError('Only original nonempty64 selected L inputs')
  return self.obj.apply(x,meta,epoch=epoch,step=step)
