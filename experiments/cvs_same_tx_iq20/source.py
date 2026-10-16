"""Native source roles, transient visible-L donors and final-only seven source views."""
from collections import Counter
import torch
from . import design as d,model as api,bank
from experiments.cvs_crossrx_satmix20.source import Observer as ParentObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng

class Observer(ParentObserver):
 def data(self,native,ctx,out):
  super().data(native,ctx,out);ds=ctx['train_loader'].dataset
  if len(ds)!=5040 or getattr(ds,'hide',False) or getattr(ds,'transform',None) is not None or getattr(ds.base,'transform',None) is not None:raise ValueError('Only actual unaugmented visible trainL dataset supplies donors')
  from cvsrffi.game_tracking.data import opaque_id
  xs=[];labels=[];rx=[];days=[];keys=[];identities=[]
  with _preserve_rng():
   for i in range(5040):
    x,y,domain,meta=ds[i];item=ds.index[i];key=tuple(int(meta[k]) for k in bank.KEYS)
    expected=(int(item.eq_i),int(item.rx_i),int(item.day_i),int(item.sig_i),int(ds.selected[i]))
    if key!=expected or int(y)!=int(item.tx_i) or int(y) not in range(6) or key[1] not in d.FIT:raise ValueError('Donor physical key or visible L label differs')
    xs.append(x);labels.append(int(y));rx.append(key[1]);days.append(key[2]);keys.append(key);identities.append(opaque_id(item))
  if set(identities)!=set(self.roles['L_s']):raise ValueError('Donors escape actual filtered native L role')
  self.bank=bank.Bank(torch.stack(xs),labels,rx,days,keys,identities);bank.install(self.bank)
  d.write(out/'IQmix_bank.json',dict(unique_L=5040,physical_ids=identities,keys=keys,labels=labels,RX=rx,day=days,same_donors=self.bank.same,cross_donors=self.bank.cross,dataset_IQ_reads=5040,not_in_model_teacher_or_checkpoint=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False))
 def epoch(self,m,ctx,rows,out,device):
  epoch=len(rows);steps=epoch*222;b=self.bank;generated=0 if self.c['arm']=='B00' else steps*128
  if m._feature_l_steps!=steps or m._joint_calls!=steps or m._joint_L_packets!=steps*1024 or m._IQmix_calls!=steps or m._IQmix_L_packets!=steps*256 or m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps:raise ValueError('Actual8forward/F+CROSS_RX/IQmix/update budget differs')
  if b.calls!=steps or sum(b.visits)!=steps*128 or b.generated!=generated or b.gathered!=generated or m._joint_As_packets or m._joint_coverage or sum(b.coverage.values())!=generated:raise ValueError('All and only legal labeled IQmix views required')
  d.write(out/'IQmix_progress.json',dict(epoch=epoch,steps=steps,L_packets=m._joint_L_packets,extra_L_packets=m._IQmix_L_packets,IQmix_generated=b.generated,donor_IQ_gathers=b.gathered,bank_unique_L=5040,As_packets=0,supcon_applied=m._supcon_applied,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model;b=self.bank
  if m._IQmix_calls!=4440 or m._IQmix_L_packets!=1136640 or self.teacher._IQmix_calls or self.teacher._joint_As_packets or b.calls!=4440 or sum(b.visits)!=568320:raise ValueError('Full additional L pair budget and no teacher IQmix required')
  result.update(status='SOURCE_SAMETX_IQMIX_E20_COMPLETE',Ag=self.c['arm']=='B10',As=False,IQmix_steps=m._IQmix_calls,IQmix_mode=self.c['iqmix']['mode'],IQmix_bank_unique_L=5040,IQmix_bank_direct_dataset_reads=5040,IQmix_generated_packets=b.generated,IQmix_donor_IQ_gathers=b.gathered,extra_L_forward_packets=m._IQmix_L_packets,total_pair_weight=1.25,teacher_IQmix_calls=0,final_native_logged_L_CE=result['final_L_CE'],final_base_L_objective=sum(v['base_total_L'] for v in m._joint_details[-222:])/222,final_extra_pair_objective=sum(v['extra_pair_loss'] for v in m._joint_details[-222:])/222)
  coverage=[dict(TX=k[0],anchor_RX=k[1],day=k[2],donor_RX=k[3],count=v) for k,v in sorted(b.coverage.items())]
  d.write(self.out/'IQmix_audit.json',dict(arm=self.c['arm'],config=self.c['iqmix'],calls=b.calls,unique_bank_L=5040,bank_dataset_reads=5040,anchor_observations=sum(b.visits),per_anchor_visits=b.visits,IQmix_generated=b.generated,donor_IQ_gathers=b.gathered,coverage=coverage,tracked_BN_module_names=list(m._IQmix_bn_names),extra_BN_updates=sum(v['extra_BN_batch_updates'] for v in m._joint_details),base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,bank_in_model_teacher_checkpoint=False,no_U_or_teacher_IQmix=True,As_packets=0,target_access=False))
  d.write(self.out/'satmix_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
