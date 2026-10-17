"""Only change the satellite input; no new loss, physical knob or deployment state."""
from contextlib import contextmanager
import torch
from . import design as d
from experiments.cvs_fullsource_concat22 import observer as base
base.d=d
class Observer(base.Observer):
 def __init__(self,c):
  super().__init__(c);self.raw_L=None;self.route_calls=0;self.route_packets=0;self.route_change_sum=0.
 def record(self,role,batch):
  if role=='L_s':
   x,y,domain,meta=batch
   if self.raw_L is not None or x.shape!=(128,2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ValueError('One aligned original physical L batch required')
  super().record(role,batch)
  if role=='L_s':self.raw_L=batch[0].detach().clone()
 def expanded(self,result,clean):
  super().expanded(result,clean)
  if self.route_calls!=self.concat_calls or self.route_packets!=self.concat_calls*128:raise ValueError('Exactly one raw-practical call per L update')
  self.raw_L=None
 @contextmanager
 def native_scope(self,native):
  original=native.ConcatSatChannelAugment;owner=self
  class RawInput(original):
   def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs);apply=self.apply_fn
    def routed(x,scene,args,*,gen=None,return_meta=False):
     raw=owner.raw_L
     if raw is None or raw.shape!=x.shape or len(x)!=128 or owner.current is None:raise ValueError('No U, extra view or stale physical IQ may enter satellite route')
     raw=raw.to(device=x.device,dtype=x.dtype)
     change=float((raw-x).square().mean())
     owner.route_calls+=1;owner.route_packets+=len(raw);owner.route_change_sum+=change
     if owner.route_calls==1:d.write(owner.out/'input_route_first_batch.json',dict(clean='unchanged ordinary',satellite='original normalized raw IQ',raw_shape=list(raw.shape),ordinary_to_raw_MSE=change,physical_ID_alignment='same current L metadata',state_recomputed=True,TX_used_in_channel=False,unknown_TX_hardware_inversion=False))
     return apply(raw,scene,args,gen=gen,return_meta=return_meta)
    self.apply_fn=routed
  native.ConcatSatChannelAugment=RawInput
  try:
   with super().native_scope(native):yield
  finally:native.ConcatSatChannelAugment=original
 def finish(self):
  result=super().finish()
  if self.raw_L is not None or self.route_calls!=4440 or self.route_packets!=568320:raise ValueError('Complete E20 raw routing required')
  result.update(status='FULLSOURCE_RAW_PRACTICAL_TRAINED',input_route=dict(calls=self.route_calls,packets=self.route_packets,mean_ordinary_raw_MSE=self.route_change_sum/self.route_calls,clean_unchanged=True,default_head_unchanged=True,new_physical_parameters=False,new_loss=False))
  d.write(self.out/'result.json',result);return result
