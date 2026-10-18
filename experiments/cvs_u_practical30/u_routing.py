"""Label-blind U view routing;native strong noise is drawn for all slots before routing."""
import hashlib,json,time
from collections import Counter
import numpy as np
from . import design as d
FIELDS=('eq_i','rx_i','day_i','sig_i','base_index')
NAMESPACE='FcrossRX/UPractical/slots/v1'
def stable_seed(seed,*parts):return int.from_bytes(hashlib.sha256(json.dumps([int(seed),*parts],separators=(',',':')).encode()).digest()[:8],'little')
def select(count,epoch,step):
 if not isinstance(count,int) or count<=0 or count%4 or epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1:raise ValueError('Quarter slots and fixed cumulative context required')
 seed=stable_seed(d.ALLOCATION_SEED,NAMESPACE,epoch,step);rng=np.random.default_rng(seed);positions=sorted(rng.permutation(count)[:count//4].tolist())
 return positions,dict(slot_namespace=NAMESPACE,slot_private_seed=seed,slot_augmentation_seed=d.ALLOCATION_SEED,global_RNG_used_for_routing=False,selection_uses_TX_or_pseudo=False,selection_without_replacement=True)
def metadata(meta,count):
 if any(k not in meta for k in FIELDS):raise ValueError('Existing hidden U physical metadata required')
 values={k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k]) for k in FIELDS}
 if any(len(v)!=count for v in values.values()) or set(map(int,values['rx_i']))-set(d.RX) or set(map(int,values['day_i']))-{1,2,3} or set(map(int,values['eq_i']))!={1}:raise ValueError('Allsource U RX/day/equalized metadata differs')
 if any(k in meta for k in ('tx','tx_i','true_tx_i','tx_label','y_tx')):raise ValueError('U TX truth must remain absent')
 return values
class Router:
 def __init__(self,c,channel=None):
  self.c=c;self.mode=c['u_practical']['mode'];self.calls=0;self.packets=0;self.selected=0;self.As_packets=0;self.records=[]
  if channel is None:
   from .channel import UPractical
   channel=UPractical('train')
  self.channel=channel
 def route(self,raw,strong,meta,*,epoch,step):
  import torch
  if raw.shape!=strong.shape or raw.ndim!=3 or tuple(raw.shape[1:])!=(2,256) or raw.device!=strong.device or raw.dtype!=strong.dtype or not torch.isfinite(raw).all() or not torch.isfinite(strong).all():raise ValueError('Existing finite raw/strong hiddenU IQ required')
  if self.mode not in ('GAUSSIAN_CONTROL','QUARTER_PRACTICAL'):raise ValueError('Only fixed matched U modes')
  start=time.perf_counter();count=len(raw);values=metadata(meta,count);positions,plan=select(count,epoch,step);select_end=time.perf_counter();observed=[];simulation_seconds=0.;copy_seconds=0.;student=strong
  if self.mode=='QUARTER_PRACTICAL':
   idx=torch.tensor(positions,device=raw.device,dtype=torch.long);small={k:[values[k][i] for i in positions] for k in FIELDS};tic=time.perf_counter();satellite,observed=self.channel.apply(raw[idx],small,epoch=epoch,step=step);simulation_seconds=time.perf_counter()-tic
   tic=time.perf_counter();student=strong.clone();student[idx]=satellite;copy_seconds=time.perf_counter()-tic
   if [tuple(o['sample_id']) for o in observed]!=[tuple(int(values[k][i]) for k in FIELDS) for i in positions]:raise ValueError('U practical must retain selected raw PID order')
  ids=[tuple(int(values[k][i]) for k in FIELDS) for i in range(count)]
  record=dict(mode=self.mode,step=step-4440,epoch=epoch-20,cumulative_step=step,cumulative_epoch=epoch,U_packets=count,selected_positions=positions,Gaussian_positions=[i for i in range(count) if self.mode=='GAUSSIAN_CONTROL' or i not in positions],selected_slots=len(positions),As_packets=len(observed),sample_id_fields=list(FIELDS),sample_ids=ids,RX=[int(v) for v in values['rx_i']],day=[int(v) for v in values['day_i']],eq=[int(v) for v in values['eq_i']],duplicate_PID_occurrences=count-len(set(ids)),U_truth_available=False,raw_to_practical=True,ordinary_before_practical=False,original_strong_draw_packets=count,original_strong_std=.015,original_strong_before_overwrite=True,C_returns_original_strong_object=student is strong,teacher_qualification_before_student_view=True,student_passes=1,extra_NN=0,extra_CE_operator_calls=0,selection_cpu_wall_seconds=select_end-start,simulation_adapter_cpu_wall_seconds=simulation_seconds,replacement_copy_cpu_wall_seconds=copy_seconds,route_cpu_wall_seconds=time.perf_counter()-start,physics=[dict(selected_position=pos,scene=o['scene'],sample_id=o['sample_id'],session_id=o['session_id'],namespace=o['namespace'],state=o['state_end'],lock=o['receiver_lock_status'],signal_to_added_noise_db=o.get('signal_to_added_noise_db'),quality_snr_db=o.get('quality_snr_db')) for pos,o in zip(positions,observed)],scene_counts=dict(Counter(o['scene'] for o in observed)),**plan)
  self.calls+=1;self.packets+=count;self.selected+=len(positions);self.As_packets+=len(observed)
  return student,record
 def summary(self):return dict(mode=self.mode,calls=self.calls,U_packets=self.packets,selected_slots=self.selected,As_packets=self.As_packets,channel=self.channel.summary(),extra_NN=0,U_truth_used=False,teacher_trajectory_equality_claim=False)
