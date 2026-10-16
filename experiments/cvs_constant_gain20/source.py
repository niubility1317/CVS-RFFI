"""Reuse source-only final20 seven-scene reader; gain counters are separate."""
from . import design as d,model as api
from experiments.cvs_crossrx_satmix20.source import Observer as SharedObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(SharedObserver):
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  expected=len(rows)*222*128 if self.c['arm']=='B10' else 0
  if m._gain_packets!=expected or m._joint_As_packets or m._joint_coverage:raise ValueError('Constant-gain L coverage or As boundary differs')
  d.write(out/'gain_progress.json',dict(epoch=len(rows),steps=len(rows)*222,gain_packets=m._gain_packets,As_packets=0,L_packets=m._joint_L_packets,supcon_applied=m._supcon_applied,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);result.update(status='SOURCE_CONSTANT_GAIN_COMPLETE',Ag=self.c['arm']=='B10',As=False,gain_training_observations=self.model._gain_packets,gain_max_power_error=max(v['gain_power_error'] for v in self.model._joint_details))
  if result['gain_training_observations']!=(568320 if self.c['arm']=='B10' else 0) or result['gain_max_power_error']>1e-6:raise ValueError('Complete gain coverage required')
  d.write(self.out/'satmix_result.json',result);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
