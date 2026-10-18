"""Original full physical residual renderer; only its scene index changes."""
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
from experiments.cvs_clock_factorial30.channel import Practical as Original,serial
from experiments.cvs_clock_factorial30 import design as original_design
from experiments.cvs_crossview_supcon41.channel import tensor_iq
from .sampler import choose,weights,URBAN,SCENES

class Practical(Original):
 def __init__(self,role='train',prior=URBAN):
  super().__init__(role);self.prior=weights(prior);self.scene_counts=Counter();self.uniform_min=1.;self.uniform_max=0.;self.nominal_parity=0
  if tuple(original_design.TRAIN_SCENES)!=SCENES:raise ValueError('Actual original training scene order differs')
 def __deepcopy__(self,memo):
  other=type(self)(prior=self.prior);memo[id(self)]=other;return other
 def audit(self):
  return dict(super().audit(),training_scene_order=list(SCENES),scene_probabilities=list(self.prior),scene_sampler='private legacy choice categorical bucket + private uniform;explicit inverseCDF',extra_random_offset_is_not_original_random_stream=True,q=None,loss_or_physical_parameters_changed=False)
 def apply(self,x,meta,epoch,step):
  if type(epoch) is not int or type(step) is not int or epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1 or x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite true L IQ and cumulative E21..30 only')
  keys=('eq_i','rx_i','day_i','sig_i','base_index')
  if any(k not in meta or len(meta[k])!=len(x) for k in keys):raise ValueError('Original physical source metadata required')
  v={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in keys}
  if set(map(int,v['rx_i']))-set(original_design.FIT) or set(map(int,v['day_i']))-{1,2,3} or set(map(int,v['eq_i']))!={1}:raise ValueError('All original source RX/day/equalized roles required')
  raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);out=[];records=[];namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S'
  for i in range(len(x)):
   pid=tuple(int(v[k][i]) for k in keys);rx=int(v['rx_i'][i]);day=int(v['day_i'][i]);session=f'rx{rx}:day{day}:eq{int(v["eq_i"][i])}'
   rng=np.random.default_rng(self.ch.stable_seed(original_design.CHANNEL_SEEDS['train'],namespace,'scene',pid));scene,sampling=choose(rng,self.prior);cfg=self.configs[scene]
   seed=self.ch.stable_seed(original_design.CHANNEL_SEEDS['train'],namespace,scene,pid);hw=self.ch.receiver_for_session(cfg,original_design.CHANNEL_SEEDS['receiver'],session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('RX/day hardware changed across labels or scenes')
   normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);signal,m=self.residual.ResidualChannel(cfg,seed,hw).process(normalized)
   if not np.isfinite(signal).all() or m['channel_equalization_applied']:raise ValueError('Full residual/noEQ/finite output required;no dropping')
   self.coverage[(rx,day,scene,m['state_end'],m['receiver_lock_status'])]+=1;self.scene_counts[scene]+=1;self.uniform_min=min(self.uniform_min,sampling['scene_uniform']);self.uniform_max=max(self.uniform_max,sampling['scene_uniform']);self.nominal_parity+=1
   out.append(np.stack((signal.real,signal.imag)));records.append(serial(dict(m,scene=scene,sample_id=pid,session_id=session,namespace=namespace,**sampling)))
  self.calls+=1;self.packets+=len(x);a=np.ascontiguousarray(out,dtype=np.float32)
  return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device),records
 def summary(self):
  return dict(super().summary(),scene_probabilities=list(self.prior),scene_counts=dict(self.scene_counts),categorical_coupling_nominal_parity_count=self.nominal_parity,scene_uniform_min=self.uniform_min if self.packets else None,scene_uniform_max=self.uniform_max if self.packets else None,q=None,state_filter=False,receiver_lock_filter=False,adaptive_sampling=False)
