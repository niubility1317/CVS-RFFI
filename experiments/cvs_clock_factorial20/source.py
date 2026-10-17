"""All original source RX; fixed20 and original U, never a source-fold observer."""
from experiments.cvs_fullsource_concat20.source import Observer as ReferenceObserver,RecordingLoader
from experiments.cvs_feature_sequence50.source import train as shared_train
from . import design as d,model as api

class Observer(ReferenceObserver):
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222
  if m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*128*4 or m._supcon_calls!=n or m._supcon_applied!=max(0,ep-5)*222 or self.optimizer._sinc_physical_clip_calls!=n or self.optimizer._sinc_unit_steps!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Actual matched F/SupCon/update budget differs')
  if m._full_channel is not None and m._full_channel.packets!=n*64:raise ValueError('All S-slot satellite packets must be supervised')
  if m._clock is not None and m._clock.packets!=n*64:raise ValueError('All G-slot clock observations required')
  d.write(out/'reference_progress.json',dict(epoch=ep,steps=n,L_packets=m._full_L_packets,Ag_packets=0 if m._clock is None else m._clock.packets,As_packets=0 if m._full_channel is None else m._full_channel.packets,source_RX=list(d.FIT),source_RX_holdout=None,U_truth_read=False,target_access=False))
 def finish(self):
  result=super().finish();result['clock']=None if self.model._clock is None else self.model._clock.summary();result['factorial']=self.c['factorial'];d.write(self.out/'training_result.json',result);return result

def train(c):
 if c['baseline_reuse']:raise ValueError('Existing references are not trained or used as parents')
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
