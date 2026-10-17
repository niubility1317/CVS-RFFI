"""One original complete practical residual;private NoEQ/MMSE24 receiver choice."""
from collections import Counter
from dataclasses import asdict,replace
import math,time
import numpy as np
import torch
from . import design as d
from .selector import choice

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
  if role!='train':raise ValueError('Labeled training augmentation only')
  from experiments.cvs_phase1_overlay.model import native_modules
  native_modules()
  import leo_practical.channel as ch
  import leo_practical.residual as residual
  self.ch=ch;self.residual=residual;self.calls=0;self.packets=0;self.coverage=Counter();self.sessions={};self.receiver_counts=Counter();self.receiver_calls=0;self.receiver_steps=[];self.receiver_strata=Counter();self.equalizer_applied=0;self.channel_wall_seconds=0.
  self.configs={s:ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}
 def __deepcopy__(self,memo):
  v=type(self)();memo[id(self)]=v;return v
 def audit(self):return dict(receiver_policy=d.config(d.rows()[0])['receiver_policy'],processing_seed=d.PROCESSING_SEED,channel_module_file=self.ch.__file__,residual_module_file=self.residual.__file__,configurations={s:asdict(c) for s,c in self.configs.items()},all_original_RX=list(d.FIT),ordinary_before_satellite=False,formal_test_changed=False,native_practical_EQ_flag_not_consumer=True)
 def apply(self,x,meta,epoch,step):
  start=time.perf_counter()
  if epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1 or x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite realL IQ/cumulativeE21..30 only')
  keys=('eq_i','rx_i','day_i','sig_i','base_index')
  if any(k not in meta or len(meta[k])!=len(x) for k in keys):raise ValueError('Original physical metadata required')
  v={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in keys}
  if set(map(int,v['rx_i']))-set(d.FIT) or set(map(int,v['day_i']))-{1,2,3} or set(map(int,v['eq_i']))!={1}:raise ValueError('Original fullsource RX/day/EQ1 required')
  raw=np.asarray(x.detach().cpu().tolist(),dtype=np.float64);out=[];records=[];step_counts=Counter();namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S';applied=0
  for i in range(len(x)):
   pid=tuple(int(v[k][i]) for k in keys);rx=int(v['rx_i'][i]);day=int(v['day_i'][i]);session=f'rx{rx}:day{day}:eq{int(v["eq_i"][i])}'
   rng=np.random.default_rng(self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,'scene',pid));scene=str(rng.choice(d.TRAIN_SCENES));nominal=self.configs[scene]
   seed=self.ch.stable_seed(d.CHANNEL_SEEDS['train'],namespace,scene,pid);hw=self.ch.receiver_for_session(nominal,d.CHANNEL_SEEDS['receiver'],session);hardware=asdict(hw)
   if self.sessions.setdefault(session,hardware)!=hardware:raise ValueError('Original hardware/calibration must stay fixed perphysical session')
   selected=choice(pid,epoch,step);cfg=replace(nominal,equalization_enabled=True,equalizer_method='mmse',equalizer_delay_samples=24) if selected else nominal
   normalized,_=self.ch.normalize_input(raw[i,0]+1j*raw[i,1]);signal,m=self.residual.ResidualChannel(cfg,seed,hw).process(normalized)
   expected=bool(selected and m['receiver_lock_status']!='unlocked')
   if not np.isfinite(signal).all() or m['channel_equalization_applied']!=expected or m['tx_impairment_correction_applied'] or m['added_noise_variance_reference']!=.001:raise ValueError('Complete coupled original residual/quality/noise required;no dropping')
   if expected and (m['equalizer']['method']!='mmse' or m['equalizer_target_total_delay_samples']!=24):raise ValueError('Fixed MMSE24 only')
   self.receiver_calls+=1;self.receiver_counts[selected]+=1;step_counts[selected]+=1;applied+=int(expected);self.receiver_strata[(rx,day,scene,selected,m['state_end'],m['receiver_lock_status'],expected)]+=1
   self.coverage[(rx,day,scene,m['state_end'],m['receiver_lock_status'])]+=1;out.append(np.stack((signal.real,signal.imag)));records.append(serial(dict(m,scene=scene,sample_id=pid,session_id=session,namespace=namespace,receiver_choice=selected)))
  self.receiver_steps.append(dict(epoch=epoch,step=step,NoEQ=step_counts[0],MMSE24=step_counts[1],EQ_applied=applied));self.equalizer_applied+=applied;self.calls+=1;self.packets+=len(x);a=np.ascontiguousarray(out,dtype=np.float32);self.channel_wall_seconds+=time.perf_counter()-start
  return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device),records
 def summary(self):return dict(calls=self.calls,packets=self.packets,session_count=len(self.sessions),sessions=self.sessions,receiver_calls=self.receiver_calls,receiver_counts=dict(self.receiver_counts),receiver_steps=self.receiver_steps,receiver_strata=[dict(RX=k[0],day=k[1],scene=k[2],selected=k[3],state=k[4],lock=k[5],EQ_applied=k[6],count=n) for k,n in sorted(self.receiver_strata.items())],processing_seed=d.PROCESSING_SEED,equalizer_applied=self.equalizer_applied,channel_wall_seconds=self.channel_wall_seconds,amplitude_renormalization=False,true_hardware_and_calibration_unchanged=True,propagation_prior_unchanged=True,noise_prior_unchanged=True,unlocked_EQ_skipped=True,physical_channel_prepasses=0,complete_residual_calls=self.packets,extra_neural_forwards=0,strata=[dict(RX=k[0],day=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(self.coverage.items())],all_original_source_RX=list(d.FIT),dropped=0,confidence_filter=False,ordinary_before_satellite=False,formal_test_changed=False)
