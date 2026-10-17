"""Original final20/source sevenviews;check all As objectives are CE-only."""
from . import design as d,model as api
from experiments.cvs_scene_cycle20.source import Observer as Parent
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(Parent):
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  if m._As_pair_calls!=len(rows)*222 or any(v['As_pair_objective']!='TWO_CE' or v['As_KL_loss_weight']!=0. or v['As_KL_gradient_terms']!=0 or v['extra_pair_loss']!=v['extra_pair_CE'] for v in m._joint_details):raise ValueError('EveryAs pair must be supervisedmeanCE/noKLgradient')
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model
  if m._As_pair_calls!=4440 or self.teacher._As_pair_calls or self.reader._As_pair_calls:raise ValueError('FullL-onlyCE pairbudget required')
  result.update(status='SOURCE_AS_PAIR_CE_E20_COMPLETE',As_pair_objective='TWO_CE',As_pair_calls=m._As_pair_calls,As_KL_loss_weight=0.,As_KL_gradient_terms=0,base_RDrop_unchanged=True,detached_As_KL_diagnostic=True)
  d.write(self.out/'As_pair_CE_audit.json',dict(config=self.c['as_pair_objective'],calls=4440,As_supervised_packets=568320,As_forward_packets=1136640,teacher_As_pair_calls=0,reader_As_pair_calls=0,As_KL_loss_weight=0.,As_KL_gradient_terms=0,base_RDrop_unchanged=True,target_access=False));d.write(self.out/'satmix_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
