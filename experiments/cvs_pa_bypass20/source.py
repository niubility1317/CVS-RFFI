"""Change only native ordinary-L return IQ,then automatic final source7views."""
import gzip,json
from contextlib import contextmanager
from . import design as d,model as api
from .augmentation import Wrapper
from experiments.cvs_crossrx_satmix20.source import Observer as SharedObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(SharedObserver):
 def __init__(self,c):super().__init__(c);self.PA_records=[];self.PA_schedule=[];self.PA_factory_calls=0
 @contextmanager
 def native_scope(self,native):
  original=native.make_augmentor;configure=native.configure_augmentor_for_epoch
  def factory(*args,**kwargs):
   self.PA_factory_calls+=1
   if self.PA_factory_calls!=1:raise ValueError('One original ordinary augmentor required')
   return Wrapper(original(*args,**kwargs),self.c['arm'],self.PA_records)
  def actual_schedule(aug,base_cfg,epoch,args):
   training_epoch=len(self.PA_schedule)+1
   if args.label_epochs!=13 or epoch!=min(training_epoch,13) or len(self.PA_records)!=(training_epoch-1)*222:raise ValueError('Actual native13L+7U augmentation schedule differs')
   value=configure(aug,base_cfg,epoch,args);self.PA_schedule.append(dict(training_epoch=training_epoch,configure_epoch=epoch,state=dict(value),pa_parameters=dict(vars(aug.pa))));return value
  native.make_augmentor=factory;native.configure_augmentor_for_epoch=actual_schedule
  try:
   with super().native_scope(native):yield
  finally:native.make_augmentor=original;native.configure_augmentor_for_epoch=configure
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222
  if len(self.PA_records)!=steps or len(self.PA_schedule)!=len(rows) or self.PA_factory_calls!=1 or m._joint_As_packets or m._joint_coverage:raise ValueError('One captured ordinary L per step,no As/U alteration')
  d.write(out/'PA_actual_schedule.json',self.PA_schedule)
  d.write(out/'PA_progress.json',dict(epoch=len(rows),steps=steps,native_L_calls=len(self.PA_records),native_L_packets=steps*128,bypass_returned_packets=steps*128 if self.c['arm']=='B10' else 0,triggered=sum(r['triggered'] for r in self.PA_records),changed=sum(r['changed'] for r in self.PA_records),target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views)
  if len(self.PA_records)!=4440 or sum(v['count'] for v in self.PA_records)!=568320 or self.PA_factory_calls!=1:raise ValueError('Full native PA L coverage differs')
  with gzip.open(self.out/'PA_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in self.PA_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  result.update(status='SOURCE_PA_BYPASS_COMPLETE',Ag=False,As=False,ordinary_PA_bypass=self.c['arm']=='B10',PA_native_L_calls=len(self.PA_records),PA_native_L_packets=568320,PA_bypass_returned_packets=568320 if self.c['arm']=='B10' else 0,PA_triggered_packets=sum(r['triggered'] for r in self.PA_records),PA_changed_packets=sum(r['changed'] for r in self.PA_records),PA_capture_extra_NN_packets=0)
  d.write(self.out/'satmix_result.json',result);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
