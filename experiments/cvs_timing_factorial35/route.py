"""Route the same physical L IQ; new Ag is the fixed bounded receive-window offset."""
import torch
from .timing import transform
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
class TimingRoute:
 def __init__(self,enabled):
  if type(enabled) is not bool:raise ValueError('Explicit Ag flag required')
  self.enabled=enabled;self.calls=0;self.packets=0;self.ids=set();self.offsets=set();self.power_max=0.;self.relative_sum=0.
 def apply(self,raw,ordinary,meta,step):
  if type(step) is not int or step!=self.calls+1 or raw.shape!=(128,2,256) or ordinary.shape!=raw.shape or raw.dtype!=torch.float32 or ordinary.dtype!=raw.dtype or not torch.isfinite(raw).all() or not torch.isfinite(ordinary).all():raise ValueError('Only ordered finite physical L128 allowed')
  if any(k not in meta or meta[k].shape!=(128,) or meta[k].dtype!=torch.long for k in KEYS) or set(meta['rx_i'].tolist())-set((1,3,4,6,8)) or set(meta['day_i'].tolist())-set((1,2,3)):raise ValueError('Original physical source metadata required')
  physical_ids=['/'.join(map(str,v)) for v in zip(*(meta[k].tolist() for k in KEYS))]
  occurrences={};ids=[]
  for pid in physical_ids:
   occurrence=occurrences.get(pid,0);occurrences[pid]=occurrence+1;ids.append(pid+'/occ'+str(occurrence))
  view='trainL/epoch'+str((step-1)//222+1)+'/step'+str(step)+'/secondary'
  x=raw;info=None
  if self.enabled:
   x,info=transform(x,ids,view);self.offsets.update(info['offset_samples']);self.power_max=max(self.power_max,max(info['power_relative_error']));self.relative_sum+=sum(info['relative_L2'])
  self.calls+=1;self.packets+=128;self.ids.update(physical_ids)
  return x.to(device=ordinary.device,dtype=ordinary.dtype),dict(step=step,view=view,physical_ids=physical_ids,ids=ids,Ag=self.enabled,meta=info,ordinary_raw_MSE=float((raw.to(ordinary)-ordinary).square().mean()))
 def summary(self):return dict(enabled=self.enabled,calls=self.calls,packets=self.packets,unique_physical_L=len(self.ids),offset_count=len(self.offsets),offsets=sorted(self.offsets),max_relative_power_error=self.power_max,relative_L2_mean=self.relative_sum/max(1,self.packets),new_IQ_augment_views=self.packets if self.enabled else 0,source_or_target_stream_used=False)
