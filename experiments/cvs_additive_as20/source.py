"""Original source legality/defaulthead final evaluation, with actual8forward budget."""
from . import design as d,model as api
from experiments.cvs_crossrx_satmix20.source import Observer as ParentObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(ParentObserver):
 def epoch(self,m,ctx,rows,out,device):
  epoch=len(rows);steps=epoch*222;As=steps*128 if self.c['arm']=='D_As' else 0
  if m._feature_l_steps!=steps or m._joint_calls!=steps or m._joint_L_packets!=steps*1024 or m._additive_calls!=steps or m._additive_L_packets!=steps*256 or m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps:raise ValueError('Actual8forward/F+CROSS_RX/additive/update budget differs')
  if m._joint_As_packets!=As or sum(m._joint_coverage.values())!=As:raise ValueError('All and only additive labeled As must be supervised')
  d.write(out/'additive_progress.json',dict(epoch=epoch,steps=steps,L_packets=m._joint_L_packets,extra_L_packets=m._additive_L_packets,As_packets=As,supcon_applied=m._supcon_applied,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model
  if m._additive_calls!=4440 or m._additive_L_packets!=1136640 or self.teacher._additive_calls or self.teacher._joint_As_packets:raise ValueError('Full additional L pair budget,no extra teacher As required')
  result.update(status='SOURCE_ADDITIVE_AS_COMPLETE',As=self.c['arm']=='D_As',additive_steps=m._additive_calls,extra_L_forward_packets=m._additive_L_packets,total_pair_weight=self.c['additive']['pair_weight_sum'],teacher_additive_calls=0,final_native_logged_L_CE=result['final_L_CE'],final_base_L_objective=sum(v['base_total_L'] for v in m._joint_details[-222:])/222,final_extra_pair_objective=sum(v['extra_pair_loss'] for v in m._joint_details[-222:])/222)
  d.write(self.out/'additive_audit.json',dict(arm=self.c['arm'],config=self.c['additive'],calls=m._additive_calls,tracked_BN_module_names=list(m._additive_bn_names),extra_BN_updates=sum(v['extra_BN_batch_updates'] for v in m._joint_details),base_raw_total_weight=.75,base_ordinary_weight=.25,normal_pair_count=4,As_generated_packets=m._joint_As_packets,no_As_U_or_teacher=True))
  d.write(self.out/'satmix_result.json',result);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
