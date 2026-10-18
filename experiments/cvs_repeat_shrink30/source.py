"""Original native semi-supervision/full roles,only fixed labeled IQ slot differs."""
import torch
from . import design as d,model as api
from experiments.cvs_clock_factorial30.source import Observer as OriginalObserver
from experiments.cvs_view_moment30.source import _train
class Observer(OriginalObserver):
 def epoch(self,m,ctx,rows,out,device):
  n=len(rows)*222
  if m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*768 or m._supcon_calls!=n or m._supcon_applied!=n or self.optimizer._sinc_physical_clip_calls!=n or self.optimizer._sinc_unit_steps!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Matched6Lforwards/originalSupCon/jointbudget differs')
  if m._full_channel is not None or m._repeat_ag.packets!=n*128 or m._repeat_ag.calls!=n:raise ValueError('All labeledslots only;noAs/U augmentation')
  d.write(out/'reference_progress.json',dict(epoch=len(rows),cumulative_epoch=20+len(rows),steps=n,cumulative_steps=4440+n,L_packets=m._full_L_packets,Ag_packets=n*128 if self.c['new_Ag'] else 0,As_packets=0,operator=m._repeat_ag.summary(),source_RX=list(d.FIT),source_RX_holdout=None,U_truth_read=False,target_access=False))
 def finish(self):
  result=super().finish();result.pop('clock',None);result['IQ_operator']=self.model._repeat_ag.summary();result['stable_raw_anchor_weight']=.25;result['supervised_L_loss_scale']=1.;result['L_forwards_each_step']=6;result['common_NoAs20_parent']=True
  d.write(self.out/'training_result.json',result);return result
def train(c):
 d.validate(c);p=d.parent_proof();parent=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c);observer=Observer(c,parent);_train(c,design=d,model_api=api,observer=observer,parent=parent);return observer.finish()
