"""Original fullsource jointtrainer;log only L shadow draws without altering U."""
import torch
from . import design as d,model as api
from experiments.cvs_clock_factorial30.source import Observer as OriginalObserver
from experiments.cvs_view_moment30.source import _train
class Observer(OriginalObserver):
 def initialize(self,m,c,out):
  super().initialize(m,c,out)
  if m._full_channel is not None:m._full_channel.attach_logger(out)
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  if m._full_channel is not None:
   m._full_channel.flush();d.write(out/'shadow_progress.json',dict(epoch=len(rows),cumulative_epoch=20+len(rows),**m._full_channel.summary()))
 def finish(self):
  result=super().finish()
  if self.model._full_channel is not None:self.model._full_channel.close()
  result.update(one_shared_NoAs20_parent=True,new_IQ_Ag=False);d.write(self.out/'training_result.json',result);return result
def train(c):
 d.validate(c);p=d.parent_proof();parent=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c);observer=Observer(c,parent)
 try:_train(c,design=d,model_api=api,observer=observer,parent=parent);return observer.finish()
 finally:
  if hasattr(observer,'model') and observer.model._full_channel is not None:observer.model._full_channel.close()
