"""Train-only auxiliary callback installed after EMA deepcopy at first L step."""
from contextlib import contextmanager
import gzip,json
import torch
from . import design as d
from .tangent import pair,cap,frozen_observation_mode
from experiments.cvs_time_factorial50.observe import Observer as Telemetry

class Observer(Telemetry):
 def initialize(self,model,c,out):
  source_data=getattr(self,'data',None)
  super().initialize(model,c,out);self.out=out;self.aux_calls=0;self.data=source_data;self.log=None
 def prepare(self,ctx):
  dataset=ctx['train_loader'].dataset;members={}
  if len(dataset)!=6300:raise ValueError('Exact L6300 required')
  seen=set()
  for i in range(len(dataset)):
   x,y,domain,meta=dataset[i];key=(int(y),int(meta['rx_i']),int(meta['day_i']))
   pid=tuple(int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))
   if pid in seen:raise ValueError('Repeated L physical ID')
   seen.add(pid);members.setdefault(key,[]).append(i)
  expected={(tx,rx,day) for tx in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)}
  if set(members)!=expected or any(len(v)!=70 for v in members.values()):raise ValueError('L cell contract differs')
  self.data=(dataset,members)
 def auxiliary(self,model,step,identity_loss):
  if model is not self.model or not model.training or step!=model._feature_l_steps:raise ValueError('Auxiliary role/main-update mismatch')
  if step<=1110 or step%4:return identity_loss*0,{'tangent/active':0.}
  if self.data is None:raise ValueError('Legal L context missing')
  self.aux_calls+=1;call=self.aux_calls;day=1+(call-1)%3;layer=('t1','t2')[(call-1)%2]
  dataset,members=self.data;gen=torch.Generator().manual_seed(self.c['model_seed']+98413+call*701)
  indices=[]
  for tx in range(6):
   for rx in (1,3,4,6,8):
    pool=members[tx,rx,day];indices.extend(pool[j] for j in torch.randperm(70,generator=gen)[:2].tolist())
  samples=[dataset[i] for i in indices];device=next(model.parameters()).device
  x=torch.stack([s[0] for s in samples]).to(device);y=torch.tensor([int(s[1]) for s in samples],device=device);rx=torch.tensor([int(s[3]['rx_i']) for s in samples],device=device);days=torch.full((60,),day,device=device,dtype=torch.long)
  devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
  with torch.random.fork_rng(devices=devices),frozen_observation_mode(model):
   raw,detail=pair(model,x,y,rx,days,layer,self.c['arm'])
   weighted,budget=cap(model,identity_loss,raw,min(1.,(step-1110)/1110))
  record=dict(step=step,call=call,day=day,layer=layer,detail=detail,budget=budget,weighted_loss=float(weighted.detach()),U_truth_read=False,main_CE_unchanged=True)
  if self.log is None:self.log=gzip.open(self.out/'tangent_steps.jsonl.gz','xt',encoding='utf-8')
  self.log.write(json.dumps(record,allow_nan=False)+'\n')
  return weighted,{'tangent/active':1.,'tangent/raw_loss':raw.detach(),'tangent/weighted_loss':weighted.detach(),'tangent/coefficient':budget['detached_coefficient'],'tangent/retained_energy':detail['safe_retained_energy_fraction'],'tangent/identity_rank':detail['identity_rank'],'tangent/gradient_ratio':budget['actual_norm_ratio']}
 @contextmanager
 def native_scope(self,n):
  data_builder=n._build_ssdg_wisig_data;context=n.compute_core_losses
  def data(*args,**kwargs):
   ctx=data_builder(*args,**kwargs);self.prepare(ctx);return ctx
  def losses(*args,**kwargs):
   result=context(*args,**kwargs)
   # First main-L loss occurs after original teacher was copied. Teacher never
   # inherits this bound callback, source tensors, or counter state.
   if not hasattr(self.model,'_source_feature_auxiliary_hook'):self.model._source_feature_auxiliary_hook=self.auxiliary
   return result
  n._build_ssdg_wisig_data=data;n.compute_core_losses=losses
  try:
   with super().native_scope(n):yield
  finally:
   n._build_ssdg_wisig_data=data_builder;n.compute_core_losses=context
   if hasattr(getattr(self,'model',None),'_source_feature_auxiliary_hook'):del self.model._source_feature_auxiliary_hook
   if getattr(self,'log',None) is not None:self.log.close()
 def epoch(self,model,ctx,rows,out,device):
  super().epoch(model,ctx,rows,out,device)
  if self.log is not None:self.log.flush()
  d.write(out/'tangent_execution.json',dict(auxiliary_calls=self.aux_calls,extra_L_forward_packets=self.aux_calls*180,teacher_intervention=False,U_intervention=False,head_fit_updates=0,source_only=True))
