"""Original native data roles/U/final seven source views; matched10 labeled forwards."""
from . import design as d,model as api
from experiments.cvs_crossrx_satmix20.source import Observer as ParentObserver
from experiments.cvs_feature_sequence50.source import train as shared_train

class Observer(ParentObserver):
 def epoch(self,m,ctx,rows,out,device):
  epoch=len(rows);steps=epoch*222;n=self.c['mc2']['unique_As_views'];expected=steps*128*n;pair_obs=steps*256 if n else 0
  if m._feature_l_steps!=steps or m._joint_calls!=steps or m._joint_L_packets!=steps*1280 or m._MC2_calls!=steps or m._MC2_L_packets!=steps*512 or m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps:raise ValueError('Actual10forward/F+CROSS_RX/MC2/update budget differs')
  if m._MC2_generated!=[steps*128 if n else 0,steps*128 if n==2 else 0] or m._joint_As_packets!=expected or sum(m._joint_coverage.values())!=expected or sum(m._MC2_generated)!=expected or sum(m._MC2_coverage.values())!=expected or m._MC2_As_supervised_pair_observations!=pair_obs:raise ValueError('All and only full generated labeled As views supervised')
  d.write(out/'MC2_progress.json',dict(epoch=epoch,steps=steps,L_packets=m._joint_L_packets,extra_L_packets=m._MC2_L_packets,As_generated_packets=m._joint_As_packets,As_generated_by_slot=m._MC2_generated,As_supervised_pair_observations=m._MC2_As_supervised_pair_observations,supcon_applied=m._supcon_applied,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model
  if m._MC2_calls!=4440 or m._MC2_L_packets!=2273280 or self.teacher._MC2_calls or self.teacher._joint_As_packets or self.teacher._MC2_As_supervised_pair_observations:raise ValueError('Full extra L budget and no additional teacher views required')
  result.update(status='SOURCE_PRACTICAL_MC2_COMPLETE',Ag=False,As=self.c['arm']!='M00',MC2_steps=m._MC2_calls,MC2_mode=self.c['mc2']['mode'],extra_L_forward_packets=m._MC2_L_packets,total_pair_weight=1.25,As_generated_by_slot=m._MC2_generated,As_supervised_pair_observations=m._MC2_As_supervised_pair_observations,teacher_MC2_calls=0,final_native_logged_L_CE=result['final_L_CE'],final_base_L_objective=sum(v['base_total_L'] for v in m._joint_details[-222:])/222,final_extra_pair_mean_objective=sum(v['extra_pair_mean_loss'] for v in m._joint_details[-222:])/222)
  coverage=[dict(slot=k[0],TX=k[1],RX=k[2],day=k[3],scene=k[4],state=k[5],lock=k[6],count=v) for k,v in sorted(m._MC2_coverage.items())]
  d.write(self.out/'MC2_audit.json',dict(arm=self.c['arm'],config=self.c['mc2'],calls=m._MC2_calls,As_generated_packets=m._joint_As_packets,As_generated_by_slot=m._MC2_generated,As_supervised_pair_observations=m._MC2_As_supervised_pair_observations,coverage_by_slot=coverage,tracked_BN_module_names=list(m._MC2_bn_names),extra_BN_updates=sum(v['extra_BN_batch_updates'] for v in m._joint_details),base_raw_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,no_IQ_averaging=True,no_crossview_consistency=True,no_extra_U_or_teacher_As=True))
  d.write(self.out/'satmix_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
