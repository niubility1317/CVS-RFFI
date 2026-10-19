"""Native physics first;train-only post-AGC erasure with no renormalization."""
from copy import deepcopy
from .masking import apply_mask
from experiments.cvs_crossview_supcon41.channel import tensor_iq
class MaskedPractical:
 def __init__(self,obj,mode):self.obj=obj;self.mode=mode;self.records=[]
 def __getattr__(self,k):
  if k=='obj':raise AttributeError(k)
  return getattr(self.obj,k)
 def __deepcopy__(self,memo):
  v=type(self)(deepcopy(self.obj,memo),self.mode);memo[id(self)]=v;return v
 def apply(self,x,meta,epoch,step):
  # obj is the original cumulative-offset wrapper;localepoch/step enter once.
  view,physical=self.obj.apply(x,meta,epoch=epoch,step=step)
  if len(view)!=64 or len(physical)!=64:raise ValueError('Exactly existing As64 required')
  masked,records=apply_mask(view,physical,mode=self.mode,epoch=epoch+20,step=step+4440)
  records=[dict(r,scene=p['scene'],state=p['state_end'],lock=p['receiver_lock_status'],signal_to_added_noise_db=p.get('signal_to_added_noise_db'),RX=int(p['sample_id'][1]),day=int(p['sample_id'][2])) for r,p in zip(records,physical)]
  if any(r['length']!=61 for r in records):raise ValueError('Actual native train kernel length differs from preregistered61')
  self.records.append(dict(step=step,cumulative_step=4440+step,epoch=epoch,cumulative_epoch=20+epoch,mode=self.mode,packets=64,records=records))
  return masked,[dict(p,training_observation_erasure=r,physical_quality_before_erasure=True) for p,r in zip(physical,records)]
 def audit(self):return dict(self.obj.audit(),training_observation_erasure=dict(mode=self.mode,post_native_AGC=True,probability=.5,expected_length=61,reference=[80,160],receiver_state_recomputed=False,renormalized=False,formal_mask=False))
 def summary(self):
  records=[r for b in self.records for r in b['records']]
  return dict(self.obj.summary(),training_observation_erasure=dict(mode=self.mode,calls=len(self.records),packets=len(records),masked_packets=sum(r['enabled'] for r in records),physical_quality_before_erasure=True,renormalized=False,formal_mask=False))
