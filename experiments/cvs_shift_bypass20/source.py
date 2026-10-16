"""Only ordinary-L shift return differs; fixedfinal20 then source seven views."""
import gzip,json
from contextlib import contextmanager
from . import design as d,model as api
from .augmentation import Wrapper
from experiments.cvs_crossrx_satmix20.source import Observer as SharedObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(SharedObserver):
 def __init__(self,c):super().__init__(c);self.shift_records=[];self.shift_schedule=[];self.shift_factory_calls=0
 @contextmanager
 def native_scope(self,native):
  original=native.make_augmentor;configure=native.configure_augmentor_for_epoch
  def factory(*args,**kwargs):
   self.shift_factory_calls+=1
   if self.shift_factory_calls!=1:raise ValueError('One ordinary augmentor required')
   return Wrapper(original(*args,**kwargs),self.c['arm'],self.shift_records)
  def actual_schedule(aug,base_cfg,epoch,args):
   training_epoch=len(self.shift_schedule)+1
   if args.label_epochs!=13 or epoch!=min(training_epoch,13) or len(self.shift_records)!=(training_epoch-1)*222:raise ValueError('Actual native13L+7U shift schedule differs')
   value=configure(aug,base_cfg,epoch,args);self.shift_schedule.append(dict(training_epoch=training_epoch,configure_epoch=epoch,state=dict(value),pa_parameters=dict(vars(aug.pa))));return value
  native.make_augmentor=factory;native.configure_augmentor_for_epoch=actual_schedule
  try:
   with super().native_scope(native):yield
  finally:native.make_augmentor=original;native.configure_augmentor_for_epoch=configure
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222
  if len(self.shift_records)!=steps or len(self.shift_schedule)!=len(rows) or self.shift_factory_calls!=1 or m._joint_As_packets or m._joint_coverage:raise ValueError('One original ordinary L call per step,no As/U alteration')
  d.write(out/'shift_actual_schedule.json',self.shift_schedule)
  d.write(out/'shift_progress.json',dict(epoch=len(rows),steps=steps,native_L_calls=len(self.shift_records),native_L_packets=steps*128,bypass_returned_packets=steps*128 if self.c['arm']=='B10' else 0,shift_gate_calls=sum(v['shift_gate_calls'] for v in self.shift_records),nonzero_shift=sum(v['nonzero'] for v in self.shift_records),shift_changed=sum(v['shift_changed'] for v in self.shift_records),inspection_scalar_draws=sum(v['shift_inspection_scalar_draws'] for v in self.shift_records),target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views)
  if len(self.shift_records)!=4440 or sum(v['count'] for v in self.shift_records)!=568320 or self.shift_factory_calls!=1:raise ValueError('Full original ordinary L coverage differs')
  with gzip.open(self.out/'shift_steps.jsonl.gz','xt',encoding='utf8') as f:
   for v in self.shift_records:f.write(json.dumps(v,allow_nan=False)+'\n')
  result.update(status='SOURCE_SHIFT_BYPASS_COMPLETE',Ag=False,As=False,ordinary_shift_bypass=self.c['arm']=='B10',shift_native_L_calls=len(self.shift_records),shift_native_L_packets=568320,shift_bypass_returned_packets=568320 if self.c['arm']=='B10' else 0,shift_gate_calls=sum(v['shift_gate_calls'] for v in self.shift_records),nonzero_shift_packets=sum(v['nonzero'] for v in self.shift_records),shift_changed_packets=sum(v['shift_changed'] for v in self.shift_records),shift_inspection_scalar_draws=sum(v['shift_inspection_scalar_draws'] for v in self.shift_records),shift_capture_extra_NN_packets=0)
  d.write(self.out/'satmix_result.json',result);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
