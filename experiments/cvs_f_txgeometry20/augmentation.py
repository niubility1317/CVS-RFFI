"""Original ordinary source augmentation and default relative-budget sat concat."""
import torch
from baseline_origin_sat_view import BaselineOriginSatViewAugment
from cvsrffi.eval import apply_sat_channel_for_scenario
class Satellite:
 def __init__(self,args):
  self.args=args
  self.generator=BaselineOriginSatViewAugment(seed=2027,apply_fn=apply_sat_channel_for_scenario,schedule='1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak')
 def __call__(self,x,epoch,step):
  virtual=min(200,1+int((epoch-1)/20*200))
  with torch.no_grad():v=self.generator.transform(x,args=self.args,epoch=virtual,batch_idx=step)
  if v.x.shape!=x.shape or not torch.isfinite(v.x).all():raise FloatingPointError('Nonfinite satellite view')
  return v.x.detach(),dict(scenario=v.scenario,applied=bool(v.applied),probability=v.view_prob,virtual_epoch=virtual)
