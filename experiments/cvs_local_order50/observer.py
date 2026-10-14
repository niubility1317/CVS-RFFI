import gzip,json
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_source_layer_order import design as sd
from experiments.cvs_source_layer_order.objective import capture,relation_indices
from experiments.cvs_source_layer_order.probe import cohorts,gradient_stats

@contextmanager
def deterministic(model):
 modes=[(m,m.training) for m in model.modules()];dev=next(model.parameters()).device
 try:
  with torch.random.fork_rng(devices=[dev.index or 0] if dev.type=='cuda' else []):
   model.eval();yield
 finally:
  for m,v in modes:m.training=v

def objective(maps,q,groups):
 result={};orders=[];supcons=[]
 for name in ('t1','t2'):
  z=F.normalize(F.adaptive_avg_pool1d(maps[name],4).flatten(1),dim=1,eps=1e-4);s=z@z.T;qq=q.to(z.device);delta=s[qq[:,0],qq[:,1]]-s[qq[:,0],qq[:,2]];gap=.1-delta;order=gap.relu().mean()
  ai=torch.tensor([g[0] for g in groups],device=z.device);pi=torch.tensor([g[1] for g in groups],device=z.device);ref=torch.tensor([g[2] for g in groups],device=z.device);supcon=(torch.logsumexp(s[ai[:,None],ref]/.2,dim=1)-s[ai,pi]/.2).mean();orders.append(order);supcons.append(supcon)
  active=gap.detach()>0;endpoints=torch.unique(qq[active].flatten());result[name]=dict(triplets=len(q),active=int(active.sum()),wrong_order=int((delta.detach()<0).sum()),active_packets=len(endpoints),cohort_packets=90,ORDER=float(order.detach()),SUPCON=float(supcon.detach()))
 return torch.stack(orders).mean(),torch.stack(supcons).mean(),result

def derivative(loss,params):
 values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
 result=[torch.zeros_like(p).flatten().detach().cpu() if v is None else v.detach().flatten().cpu() for p,v in zip(params,values)]
 if any(not torch.isfinite(v).all() for v in result):raise ValueError('Nonfinite actual feature derivative')
 return result

class Observer:
 def initialize(self,model,c,out):
  from experiments.cvs_nocfo_foundation.source import load_source
  from experiments.cvs_nocfo_foundation import design as nd
  d.validate(c);self.c=c;self.model=model;self.out=out;self.steps=0;self.extra=0;self.U_steps=0;self.U_packets=0;self.U_accept=0;self.records=[];self.saved=[];self.grad=[]
  L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;self.L=L;self.ix=cohorts(L)
  if sd.SEED+118011!=d.COHORT_SEED:raise ValueError('Fixed cohort seed differs')
  first=self.ix[0];self.q,self.groups=relation_indices(*(L[k][first] for k in ('y','rx','day')),[L['ids'][i] for i in first.tolist()]);self.params=[dict(model.named_parameters())[n] for n in sd.PARAMETERS];self.head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
  if not all(p.requires_grad for p in self.params+[self.head]):raise ValueError('Default feature/head joint training differs')
  self.stream=gzip.open(out/'local_order_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3)
 @contextmanager
 def native_scope(self,n):
  if n._source_feature_auxiliary_hook is not None or n._trusted_pseudo_loss_hook is not None:raise ValueError('Another auxiliary owner active')
  n._source_feature_auxiliary_hook=self.aux;n._trusted_pseudo_loss_hook=self.base
  try:yield
  finally:n._source_feature_auxiliary_hook=None;n._trusted_pseudo_loss_hook=None;self.stream.close()
 def base(self,*,model,args,epoch,batch_idx,loss_l,loss_u,loss_ent,active,logits,pseudo,mask):
  step=(epoch-1)*222+batch_idx
  if model is not self.model or step!=self.steps+1 or bool(active)!=(epoch>=34) or args.lambda_u!=.16 or args.lambda_ent!=.01:raise ValueError('Original source/U recipe differs')
  self.steps=step;self.main_CE=loss_l
  if active:self.U_steps+=1;self.U_packets+=len(pseudo);self.U_accept+=int(mask.sum())
  return loss_u
 def aux(self,model,step):
  if model is not self.model or step!=self.steps:raise ValueError('Source auxiliary step differs')
  epoch=(step-1)//222+1;batch=(step-1)%222+1
  if batch%4:return self.main_CE*0,dict(local_order_extra_forward=0.)
  rank=self.extra%70;ix=self.ix[rank];self.extra+=1;dev=self.params[0].device
  with deterministic(model),capture(model) as maps:out=model.forward_identity_only(self.L['x'][ix].to(dev))
  order,supcon,stats=objective(maps,self.q,self.groups);ramp=min(1.,max(0.,(epoch-5)/10));mode=self.c['local_order']['mode'];selected=order if mode!='SUPCON' else supcon;weight=0. if mode=='ZERO' else .1*ramp;effective=weight*selected
  if torch.autograd.grad(selected,self.head,allow_unused=True,retain_graph=True)[0] is not None:raise ValueError('Auxiliary depends on classW')
  if not torch.isfinite(effective):raise ValueError('Nonfinite auxiliary objective')
  record=dict(step=step,epoch=epoch,batch=batch,cohort_rank=rank,mode=mode,ramp=ramp,coefficient=weight,raw_order=float(order.detach()),raw_supcon=float(supcon.detach()),effective_loss=float(effective.detach()),source_only=True,extra_source_packets=90,layers=stats)
  if epoch in d.DIAGNOSTIC_EPOCHS:
   vectors=dict(main_CE=derivative(self.main_CE,self.params),raw_ORDER=derivative(.1*order,self.params),raw_SUPCON=derivative(.1*supcon,self.params),effective=derivative(effective,self.params));self.grad.append(dict(step=step,epoch=epoch,vectors=vectors));details=[]
   for layer,name in enumerate(sd.PARAMETERS):details.append(dict(parameter=name,ORDER=gradient_stats(vectors['main_CE'][layer],vectors['raw_ORDER'][layer]),SUPCON=gradient_stats(vectors['main_CE'][layer],vectors['raw_SUPCON'][layer]),effective=gradient_stats(vectors['main_CE'][layer],vectors['effective'][layer])))
   self.records.append(dict(step=step,epoch=epoch,layers=details))
  self.stream.write(json.dumps(record,allow_nan=False)+chr(10));self.main_CE=None
  return effective,dict(local_order_extra_forward=1.,local_order_raw=record['raw_order'],local_order_supcon=record['raw_supcon'],local_order_effective=record['effective_loss'])
 def epoch(self,model,ctx,rows,out,device):
  epoch=int(rows[-1]['epoch']);self.stream.flush()
  if self.steps!=epoch*222 or self.extra!=epoch*55:raise ValueError('Actual native/extra exposure incomplete')
  if epoch in d.DIAGNOSTIC_EPOCHS:
   if len(self.grad)!=55 or any(v['epoch']!=epoch for v in self.grad):raise ValueError('Declared gradient epoch incomplete')
   p=out/f'local_gradients_e{epoch}.pt';torch.save(self.grad,p);saved=torch.load(p,map_location='cpu',weights_only=False);records=[v for v in self.records if v['epoch']==epoch]
   for a,b in zip(saved,records):
    for k,n in enumerate(sd.PARAMETERS):
     for mode,key in [('raw_ORDER','ORDER'),('raw_SUPCON','SUPCON'),('effective','effective')]:
      if gradient_stats(a['vectors']['main_CE'][k],a['vectors'][mode][k])!=b['layers'][k][key]:raise ValueError('Saved actual gradient recount differs')
   self.saved.append(epoch);self.grad=[];d.write(out/'local_gradient_records.json',dict(status='VERIFIED',epochs=self.saved,batch_records=self.records,no_extra_head=True,U_truth_read=False,saved_gradient_recount='VERIFIED',scope='actualmainL128CE versus scheduled separateL90aux; no cumulative optimizer contribution claim'))
 def finish(self):
  if self.steps!=11100 or self.extra!=2750 or self.U_steps!=3774 or self.U_packets!=963900 or self.saved!=list(d.DIAGNOSTIC_EPOCHS) or len(self.records)!=220:raise ValueError('Native/auxiliary/gradient budget incomplete')
  value=dict(status='VERIFIED',steps=self.steps,extra_batches=self.extra,extra_source_packet_forwards=self.extra*90,U_steps=self.U_steps,U_packets=self.U_packets,U_accepted=self.U_accept,gradient_epochs=self.saved,gradient_batches=220,layer_gradient_records=660,head_fit_updates=0,auxiliary_classW_dependency=False,default_head_jointly_trained=True,U_truth_read=False,source_only=True,saved_gradient_recount='VERIFIED')
  d.write(self.out/'local_training_observation.json',value);return value
