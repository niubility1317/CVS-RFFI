"""Native roles/U and final-only seven sourceviews;no target construction."""
from . import design as d,model as api
from experiments.cvs_crossrx_satmix20.source import Observer as ParentObserver
from experiments.cvs_feature_sequence50.source import train as shared_train

class Observer(ParentObserver):
 def epoch(self,m,ctx,rows,out,device):
  epoch=len(rows);steps=epoch*222;expected=steps*128 if self.c['arm']=='B10' else 0
  if m._feature_l_steps!=steps or m._joint_calls!=steps or m._joint_L_packets!=steps*1024 or m._noise_calls!=steps or m._noise_L_packets!=steps*256 or m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps:raise ValueError('Actual8forward/F+CROSS_RX/noise/update budget differs')
  if m._noise_generated!=expected or sum(m._noise_coverage.values())!=expected or m._joint_As_packets or m._joint_coverage:raise ValueError('All and only labeled noise view coverage required')
  d.write(out/'noise_progress.json',dict(epoch=epoch,steps=steps,L_packets=m._joint_L_packets,extra_L_packets=m._noise_L_packets,noise_generated=m._noise_generated,As_packets=0,supcon_applied=m._supcon_applied,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model
  if m._noise_calls!=4440 or m._noise_L_packets!=1136640 or self.teacher._noise_calls or self.teacher._noise_generated or self.teacher._joint_As_packets:raise ValueError('Full extra L budget,no teacher newviews required')
  result.update(status='SOURCE_ISOLATED_AWGN_E20_COMPLETE',Ag=self.c['arm']=='B10',As=False,noise_steps=m._noise_calls,noise_mode=self.c['isolated_noise']['mode'],noise_generated_packets=m._noise_generated,extra_L_forward_packets=m._noise_L_packets,total_pair_weight=1.25,teacher_noise_calls=0,final_native_logged_L_CE=result['final_L_CE'],final_base_L_objective=sum(v['base_total_L'] for v in m._joint_details[-222:])/222,final_extra_pair_objective=sum(v['extra_pair_loss'] for v in m._joint_details[-222:])/222)
  coverage=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(m._noise_coverage.items())]
  d.write(self.out/'noise_audit.json',dict(arm=self.c['arm'],config=self.c['isolated_noise'],calls=m._noise_calls,generated_packets=m._noise_generated,coverage=coverage,tracked_BN_module_names=list(m._noise_bn_names),extra_BN_updates=sum(v['extra_BN_batch_updates'] for v in m._joint_details),base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,all_tails_retained=True,no_U_or_teacher_noise=True,As_packets=0,target_access=False));d.write(self.out/'satmix_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
