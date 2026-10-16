"""Same fullpractical IQ map,only training sceneassignment differs."""
import time
from collections import Counter
import numpy as np
import torch
from . import design as d
from experiments.cvs_crossrx_satmix20.channel import PairedChannel
class SceneChannel(PairedChannel):
 def __init__(self,mode):
  if mode not in ('none','iid','balanced'):raise ValueError('Fixed scene scheduling only')
  super().__init__();self.mode=mode;self.visits=Counter();self.permutations={};self.counts={};self.trace=[];self.training_wall_seconds=0.
 def __deepcopy__(self,memo):
  obj=type(self)(self.mode);memo[id(self)]=obj;return obj
 def audit(self):
  return dict(super().audit(),scene_schedule_mode=self.mode,training_only=True,permutation_seed=d.PERMUTATION_SEED,evaluation_original_unchanged=True)
 def choose(self,pid,namespace):
  # Keep the original private IID draw in all schedules;doesnot touch global RNG.
  rng=np.random.default_rng(self.ch.stable_seed(20261016101,namespace,'scene',pid));iid=int(rng.integers(6));v=self.visits[pid]
  if self.mode=='balanced':
   if v%6==0:self.permutations[pid]=np.random.default_rng(self.ch.stable_seed(d.PERMUTATION_SEED,'physical_scene_cycle',pid,v//6)).permutation(6).tolist()
   selected=int(self.permutations[pid][v%6])
  elif self.mode=='iid':selected=iid
  else:raise ValueError('No training channel permitted for rawcontrol')
  self.visits[pid]+=1;self.counts.setdefault(pid,[0]*6)[selected]+=1;return selected,v
 def apply(self,x,meta,*,namespace,scene=None):
  if namespace.startswith('source_eval/'):return super().apply(x,meta,namespace=namespace,scene=scene)
  if not namespace.startswith('train/epoch') or scene is not None or self.mode=='none':raise ValueError('Only registered L training namespace,never forced scene')
  if x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite raw IQ256 required')
  required=('rx_i','day_i','eq_i','sig_i','base_index')
  if not all(k in meta and len(meta[k])==len(x) for k in required):raise ValueError('Actual physicalL metadata required')
  values={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in required}
  if any(int(r) not in d.FIT for r in values['rx_i']) or any(int(day) not in (1,2,3) for day in values['day_i']):raise ValueError('FitL RX/day only')
  started=time.perf_counter();raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);outputs=[];details=[];indices=[];visit=[]
  for i in range(len(x)):
   pid=tuple(int(values[k][i]) for k in required);selected,v=self.choose(pid,namespace);name=d.SCENES[selected];cfg=self.configs[name];session='rx%d:day%d:eq%d'%(pid[0],pid[1],pid[2]);seed=self.ch.stable_seed(20261016101,namespace,name,pid);receiver=self.ch.receiver_for_session(cfg,20261016102,session)
   record,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);out,m=self.residual.ResidualChannel(cfg,seed,receiver).process(record)
   if not np.isfinite(out).all():raise FloatingPointError('Invalid fullpractical output;no packet dropped')
   outputs.append(np.stack([out.real,out.imag]));indices.append(selected);visit.append(v);details.append(dict(scene=name,seed=seed,session=session,RX=pid[0],day=pid[1],state=m['state_end'],lock=m['receiver_lock_status'],quality_snr_db=m['quality_snr_db'],signal_to_added_noise_db=m['signal_to_added_noise_db'],agc_limited=m['agc_limited']))
  array=np.ascontiguousarray(outputs,dtype=np.float32);result=torch.frombuffer(bytearray(array.tobytes()),dtype=torch.float32).reshape(array.shape).to(x.device)
  self.training_wall_seconds+=time.perf_counter()-started;self.trace.append(dict(namespace=namespace,scene_indices=indices,visits_before=visit))
  return result,details
 def summary(self):
  return dict(mode=self.mode,training_calls=len(self.trace),training_packets=sum(self.visits.values()),unique_physical_L=len(self.visits),max_minus_min=max((max(v)-min(v) for v in self.counts.values()),default=0),training_generation_wall_seconds=self.training_wall_seconds,permutation_cycles_constructed=sum((v+5)//6 for v in self.visits.values()) if self.mode=='balanced' else 0)
