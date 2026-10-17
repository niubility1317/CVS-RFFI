"""All original loaders/U recipe; raw secondary counterfactual in each fixed row."""
import gzip,json,sys
from contextlib import contextmanager
import torch
from . import design as d
from .route import TimingRoute
from experiments.cvs_fullsource_concat30_25 import observer as base
base.d=d
class Observer(base.Observer):
 def __init__(self,c):
  super().__init__(c);self.raw_L=None;self.secondary=None;self.route=TimingRoute(c['Ag'])
 def record(self,role,batch):
  if role=='L_s' and (self.raw_L is not None or batch[0].shape!=(128,2,256)):raise ValueError('Exactly one physical L batch pending')
  super().record(role,batch)
  if role=='L_s':self.raw_L=batch[0].detach().clone()
 def data(self,native,ctx,out):
  super().data(native,ctx,out);self.route_log=gzip.open(self.out/'timing_views.jsonl.gz','xt',encoding='utf8',compresslevel=3)
 def initialize(self,m,c,out):
  super().initialize(m,c,out);files=d.read(self.out/'module_files.json');import leo_practical.residual as residual
  files['leo_practical.residual']=residual.__file__;files['bounded_timing']=sys.modules['experiments.cvs_timing_factorial35.timing'].__file__;d.write(self.out/'module_files.json',files)
 def expanded(self,result,clean):
  if self.current is None or self.secondary is None or self.raw_L is None:raise ValueError('Aligned physical secondary view missing')
  labels,meta=self.current
  if result.x.shape!=(256,2,256) or not torch.equal(result.x[:128],clean) or not torch.equal(result.y[:128],result.y[128:]) or result.applied!=self.c['As'] or (not self.c['As'] and not torch.equal(result.x[128:],self.secondary)):raise ValueError('Fixed primary/secondary/default labels differ')
  self.concat_calls+=1
  if self.route.calls!=self.concat_calls:raise ValueError('One raw route per physical L step required')
  if self.c['As']:
   from cvsrffi import practical_adapter as pa
   records=pa._last_meta['records']
   if len(records)!=128 or pa._last_meta['config']['boundary']!='reflect' or any(r['channel_equalization_applied'] or r['tx_impairment_correction_applied'] for r in records):raise ValueError('Original reflect/G1/no TX inversion required')
   if any(r['residual_kernel_length']!=62 or r['numerical_filter_latency_samples']!=24 for r in records):raise ValueError('Original full-G1 memory changed;do not drop samples')
   self.sat_packets+=128
   for t,rx,day,r in zip(labels,meta['rx_i'].tolist(),meta['day_i'].tolist(),records):self.sat_coverage[(t,rx,day,result.scenario,r['state_end'],r['receiver_lock_status'])]+=1
   if self.last_physics is None:self.last_physics=pa._last_meta;d.write(self.out/'actual_practical_first_batch.json',pa._last_meta)
   self.physics.write(json.dumps(dict(step=self.concat_calls,scene=result.scenario,records=records))+'\n')
  self.current=None;self.raw_L=None;self.secondary=None
 @contextmanager
 def native_scope(self,native):
  original=native.ConcatSatChannelAugment;owner=self
  class RawSecondary(original):
   def expand(self,x,y,d_raw,*,args,epoch,batch_idx):
    step=owner.concat_calls+1
    if owner.current is None or owner.raw_L is None or epoch!=(step-1)//222+1 or batch_idx!=(step-1)%222+1 or y.dtype!=torch.long or y.shape!=(128,) or y.tolist()!=owner.current[0]:raise ValueError('Only real ordered labeled L may use secondary Ag/As')
    modified,log=owner.route.apply(owner.raw_L,x,owner.current[1],step);owner.route_log.write(json.dumps(log,allow_nan=False)+'\n');owner.secondary=modified
    result=super().expand(modified,y,d_raw,args=args,epoch=epoch,batch_idx=batch_idx)
    if not torch.equal(result.x[:128],modified):raise ValueError('Native first half did not retain its input')
    result.x=torch.cat((x,result.x[128:]),0);return result
  native.ConcatSatChannelAugment=RawSecondary
  try:
   with super().native_scope(native):yield
  finally:
   native.ConcatSatChannelAugment=original
   if hasattr(self,'route_log'):self.route_log.close()
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  if self.route.calls!=len(rows)*222 or self.route.packets!=self.route.calls*128 or self.raw_L is not None or self.secondary is not None:raise ValueError('Complete raw-secondary path required')
  self.route_log.flush();self.epochs[-1]['timing_route']=self.route.summary();d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
 def finish(self):
  result=super().finish()
  if self.route.calls!=6660 or self.route.packets!=852480 or len(self.route.ids)!=6300 or (self.c['Ag'] and len(self.route.offsets)!=33):raise ValueError('Full L/epoch/all33 timing offsets required')
  result.update(status='FULLSOURCE_TIMING_FACTORIAL_TRAINED',Ag=self.c['Ag'],As=self.c['As'],timing_route=self.route.summary(),primary_ordinary_preserved=True,secondary_raw_counterfactual=True)
  d.write(self.out/'result.json',result);return result
