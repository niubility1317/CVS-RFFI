"""Original fullpractical downstream;only within-call scenario counts differ."""
from collections import Counter
from dataclasses import asdict
import numpy as np
import torch
from experiments.cvs_clock_factorial30 import channel as original
from experiments.cvs_crossview_supcon41.channel import tensor_iq
from .assignment import allocation,NAMESPACE
from .protocol import cumulative
d=original.d
serial=original.serial
class Practical(original.Practical):
 def __init__(self,role='train'):
  super().__init__(role);self.scene_balance_records=[]
 def audit(self):return dict(super().audit(),scene_assignment='original64 positions21/21/22,rotating extra',private_assignment_namespace=NAMESPACE,extra_NN=0,physical_ID_uniqueness_required=False)
 def apply(self,x,meta,epoch,step):
  if epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1 or x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite real L training IQ/declared cumulative E21..30 only')
  keys=('eq_i','rx_i','day_i','sig_i','base_index')
  if any(k not in meta or len(meta[k])!=len(x) for k in keys):raise ValueError('Physical source metadata required')
  v={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in keys}
  if set(map(int,v['rx_i']))-set(d.FIT) or set(map(int,v['day_i']))-{1,2,3} or set(map(int,v['eq_i']))!={1}:raise ValueError('Original full source RX/day/equalized roles required')
  raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);out=[];records=[];namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S'
  plan=allocation(step,namespace,d.CHANNEL_SEEDS['train'],self.ch.stable_seed,count=len(x));packet_seeds=[]
  for i in range(len(x)):
   pid=tuple(int(v[k][i]) for k in keys);rx=int(v['rx_i'][i]);day=int(v['day_i'][i]);session=f'rx{rx}:day{day}:eq{int(v["eq_i"][i])}'
   scene=d.TRAIN_SCENES[plan['scene_indices'][i]];cfg=self.configs[scene]
   seed=self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,scene,pid);hw=self.ch.receiver_for_session(cfg,d.CHANNEL_SEEDS['receiver'],session);hardware=asdict(hw)
   packet_seeds.append(int(seed))
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('Hardware changed within label-independent physical session')
   normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);signal,m=self.residual.ResidualChannel(cfg,seed,hw).process(normalized)
   if not np.isfinite(signal).all() or m['channel_equalization_applied']:raise ValueError('Original full residual/noEQ/finite output required;no dropping')
   self.coverage[(rx,day,scene,m['state_end'],m['receiver_lock_status'])]+=1;out.append(np.stack((signal.real,signal.imag)));records.append(serial(dict(m,scene=scene,sample_id=pid,session_id=session,namespace=namespace)))
  self.calls+=1;self.packets+=len(x);a=np.ascontiguousarray(out,dtype=np.float32)
  counts=Counter(o['scene'] for o in records);self.scene_balance_records.append(dict(step=step-4440,cumulative_step=step,epoch=epoch-20,cumulative_epoch=epoch,namespace=namespace,train_seed=d.CHANNEL_SEEDS['train'],assignment_namespace=NAMESPACE,**plan,input_positions=list(range(64)),selected_packets=64,As_packets=64,scene0_packets=counts[d.TRAIN_SCENES[0]],scene1_packets=counts[d.TRAIN_SCENES[1]],scene2_packets=counts[d.TRAIN_SCENES[2]],scene_counts=dict(counts),sample_id_fields=['eq_i','rx_i','day_i','sig_i','base_index'],sample_ids=[o['sample_id'] for o in records],RX=[int(v['rx_i'][i]) for i in range(64)],day=[int(v['day_i'][i]) for i in range(64)],state=[o['state_end'] for o in records],lock=[o['receiver_lock_status'] for o in records],session_ids=[o['session_id'] for o in records],channel_seeds=packet_seeds,channel_seed_formula='stable_seed(train_seed,namespace,assignedscene,pid)',physical_ID_uniqueness_required=False,sampler_changed=False,global_RNG_used=False,extra_NN=0,new_losses=0,formal_test_changed=False))
  return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device),records
 def summary(self):
  value=super().summary();totals=Counter()
  for r in self.scene_balance_records:totals.update(r['scene_counts'])
  return dict(value,scene_balance=dict(calls=len(self.scene_balance_records),scene_counts=dict(totals),extra_scene_phase='(cumulative_step-1)%3',assignment_namespace=NAMESPACE,selected_packets_per_call=64,extra_NN=0,physical_ID_uniqueness_required=False))
class Shifted:
 def __init__(self,obj):self.obj=obj
 def __getattr__(self,k):
  if k=='obj':raise AttributeError(k)
  return getattr(self.obj,k)
 def apply(self,x,meta,epoch,step):
  epoch,step=cumulative(epoch,step)
  if len(x)!=64:raise ValueError('Only unchanged actual64 selected L satellite inputs')
  return self.obj.apply(x,meta,epoch=epoch,step=step)
