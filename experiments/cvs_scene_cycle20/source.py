"""Original source legality/defaulthead final evaluation, with actual8forward budget."""
import gzip,json
from . import design as d,model as api
from experiments.cvs_crossrx_satmix20.source import Observer as ParentObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(ParentObserver):
 def epoch(self,m,ctx,rows,out,device):
  epoch=len(rows);steps=epoch*222;As=steps*128 if self.c['scene_schedule']['mode']!='none' else 0
  if m._feature_l_steps!=steps or m._joint_calls!=steps or m._joint_L_packets!=steps*1024 or m._additive_calls!=steps or m._additive_L_packets!=steps*256 or m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps:raise ValueError('Actual8forward/F+CROSS_RX/additive/update budget differs')
  if m._joint_As_packets!=As or sum(m._joint_coverage.values())!=As:raise ValueError('All and only additive labeled As must be supervised')
  summary=m._joint_channel.summary()
  if summary['training_packets']!=As or summary['training_calls']!=(steps if As else 0) or (self.c['scene_schedule']['mode']=='balanced' and summary['max_minus_min']>1):raise ValueError('Actual scene coverage budget differs')
  d.write(out/'scene_cycle_progress.json',dict(epoch=epoch,steps=steps,L_packets=m._joint_L_packets,extra_L_packets=m._additive_L_packets,As_packets=As,supcon_applied=m._supcon_applied,scene_schedule=summary,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model
  if m._additive_calls!=4440 or m._additive_L_packets!=1136640 or self.teacher._additive_calls or self.teacher._joint_As_packets:raise ValueError('Full additional L pair budget,no extra teacher As required')
  result.update(status='SOURCE_ADDITIVE_AS_COMPLETE',As=self.c['scene_schedule']['mode']!='none',additive_steps=m._additive_calls,extra_L_forward_packets=m._additive_L_packets,total_pair_weight=self.c['additive']['pair_weight_sum'],teacher_additive_calls=0,final_native_logged_L_CE=result['final_L_CE'],final_base_L_objective=sum(v['base_total_L'] for v in m._joint_details[-222:])/222,final_extra_pair_objective=sum(v['extra_pair_loss'] for v in m._joint_details[-222:])/222)
  d.write(self.out/'additive_audit.json',dict(arm=self.c['arm'],config=self.c['additive'],calls=m._additive_calls,tracked_BN_module_names=list(m._additive_bn_names),extra_BN_updates=sum(v['extra_BN_batch_updates'] for v in m._joint_details),base_raw_total_weight=.75,base_ordinary_weight=.25,normal_pair_count=4,As_generated_packets=m._joint_As_packets,no_As_U_or_teacher=True))
  channel=m._joint_channel;summary=channel.summary();expected=0 if self.c['scene_schedule']['mode']=='none' else 568320
  if summary['training_packets']!=expected or summary['training_calls']!=(4440 if expected else 0) or (expected and summary['unique_physical_L']!=5040) or self.teacher._joint_channel.summary()['training_calls'] or self.reader._joint_channel.summary()['training_calls']:raise ValueError('Actual L-only scene schedule budget differs')
  if self.c['scene_schedule']['mode']=='balanced' and summary['max_minus_min']>1:raise ValueError('Sixscene perpacket balance violated')
  result.update(status='SOURCE_SCENE_CYCLE_E20_COMPLETE',scene_schedule_mode=self.c['scene_schedule']['mode'],scene_schedule_training_calls=summary['training_calls'],balanced_coverage_max_minus_min=summary['max_minus_min'],channel_generation_wall_seconds=summary['training_generation_wall_seconds'])
  with gzip.open(self.out/'scene_cycle_steps.jsonl.gz','xt',encoding='utf8') as f:
   for row in channel.trace:f.write(json.dumps(row,allow_nan=False)+'\n')
  d.write(self.out/'scene_cycle_audit.json',dict(config=self.c['scene_schedule'],summary=summary,per_packet_counts=[dict(pid=list(pid),scene_counts=counts) for pid,counts in sorted(channel.counts.items())],all_legal_L_supervised=True,source_eval_unchanged=True,teacher_training_calls=0,reader_training_calls=0,target_access=False))
  d.write(self.out/'satmix_result.json',result);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
