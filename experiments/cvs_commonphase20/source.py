"""Reuse source-only final20 seven-scene reader; phase counters are separate."""
from . import design as d,model as api
from experiments.cvs_crossrx_satmix20.source import Observer as SharedObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(SharedObserver):
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  expected=len(rows)*222*128 if self.c['arm']=='B10' else 0
  if m._phase_packets!=expected or m._joint_As_packets or m._joint_coverage:raise ValueError('Phase-only L coverage or As boundary differs')
  d.write(out/'phase_progress.json',dict(epoch=len(rows),steps=len(rows)*222,phase_packets=m._phase_packets,As_packets=0,L_packets=m._joint_L_packets,supcon_applied=m._supcon_applied,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);result.update(status='SOURCE_PHASE_COMPLETE',Ag=self.c['arm']=='B10',As=False,phase_training_observations=self.model._phase_packets,phase_max_power_error=max(v['phase_power_error'] for v in self.model._joint_details))
  if result['phase_training_observations']!=(568320 if self.c['arm']=='B10' else 0) or result['phase_max_power_error']>1e-6:raise ValueError('Complete phase coverage required')
  d.write(self.out/'satmix_result.json',result);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
