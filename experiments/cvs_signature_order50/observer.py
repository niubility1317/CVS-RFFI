import gzip,json
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_local_order50.observer import deterministic,derivative
from experiments.cvs_source_layer_order.objective import relation_indices
from experiments.cvs_source_layer_order.probe import cohorts,gradient_stats
from experiments.cvs_source_signature_order.objective import capture,descriptors

def objective(maps,q):
 s=maps['SIG_DESCRIPTOR'];dynamic,clock=descriptors(s);result={};losses={}
 for name,v in [('SIG_RAW',s),('SIG_DYNAMIC',dynamic)]:
  z=F.normalize(v,dim=1,eps=1e-4);similarity=z@z.T;qq=q.to(z.device);delta=similarity[qq[:,0],qq[:,1]]-similarity[qq[:,0],qq[:,2]];gap=.1-delta;loss=gap.relu().mean();losses[name]=loss;active=gap.detach()>0
  result[name]=dict(triplets=len(q),active=int(active.sum()),wrong_order=int((delta.detach()<0).sum()),active_packets=len(torch.unique(qq[active].flatten())),cohort_packets=90,hinge=float(loss.detach()))
 return losses,result,dict(clock_min=float(clock.detach().min()),clock_max=float(clock.detach().max()),mean_clock_energy_fraction=float((clock.square().sum(1)/s.square().sum(1).clamp_min(1e-12)).detach().mean()))

class Observer:
 def initialize(self,model,c,out):
  from experiments.cvs_nocfo_foundation.source import load_source
  from experiments.cvs_nocfo_foundation import design as nd
  d.validate(c);self.c=c;self.model=model;self.out=out;self.steps=0;self.extra=0;self.U_steps=0;self.U_packets=0;self.U_accept=0;self.records=[];self.saved=[];self.grad=[]
  L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;self.L=L;self.ix=cohorts(L);first=self.ix[0]
  self.q,self.groups=relation_indices(*(L[k][first] for k in ('y','rx','day')),[L['ids'][i] for i in first.tolist()]);self.params=[dict(model.named_parameters())[n] for n in d.PARAMETERS];self.head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
  if not all(p.requires_grad for p in self.params+[self.head]) or len(self.q)!=1800:raise ValueError('Default source feature/head relation contract differs')
  self.stream=gzip.open(out/'signature_order_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3)
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
  if batch%4:return self.main_CE*0,dict(signature_order_extra_forward=0.)
  rank=self.extra%70;ix=self.ix[rank];self.extra+=1;dev=self.params[0].device
  with deterministic(model),capture(model) as maps:out=model.forward_identity_only(self.L['x'][ix].to(dev))
  losses,stats,clock=objective(maps,self.q);ramp=min(1.,max(0.,(epoch-5)/10));mode=self.c['signature_order']['mode'];selected=losses['SIG_RAW' if mode=='ZERO' else mode];weight=0. if mode=='ZERO' else .1*ramp;effective=weight*selected
  if any(torch.autograd.grad(loss,self.head,allow_unused=True,retain_graph=True)[0] is not None for loss in losses.values()):raise ValueError('Auxiliary depends on classW')
  if not torch.isfinite(effective):raise ValueError('Nonfinite auxiliary objective')
  record=dict(step=step,epoch=epoch,batch=batch,cohort_rank=rank,mode=mode,ramp=ramp,coefficient=weight,raw_losses={k:float(v.detach()) for k,v in losses.items()},effective_loss=float(effective.detach()),source_only=True,extra_source_packets=90,views=stats,clock=clock)
  if epoch in d.DIAGNOSTIC_EPOCHS:
   vectors=dict(main_CE=derivative(self.main_CE,self.params),effective=derivative(effective,self.params));vectors.update({name:derivative(.1*loss,self.params) for name,loss in losses.items()});self.grad.append(dict(step=step,epoch=epoch,vectors=vectors));details=[]
   for layer,name in enumerate(d.PARAMETERS):details.append(dict(parameter=name,**{key:gradient_stats(vectors['main_CE'][layer],vectors[key][layer]) for key in (*d.LEVELS,'effective')}))
   self.records.append(dict(step=step,epoch=epoch,layers=details))
  self.stream.write(json.dumps(record,allow_nan=False)+chr(10));self.main_CE=None
  return effective,dict(signature_order_extra_forward=1.,signature_order_raw=record['raw_losses']['SIG_RAW'],signature_order_dynamic=record['raw_losses']['SIG_DYNAMIC'],signature_order_effective=record['effective_loss'])
 def epoch(self,model,ctx,rows,out,device):
  epoch=int(rows[-1]['epoch']);self.stream.flush()
  if self.steps!=epoch*222 or self.extra!=epoch*55:raise ValueError('Actual native/extra exposure incomplete')
  if epoch in d.DIAGNOSTIC_EPOCHS:
   if len(self.grad)!=55 or any(v['epoch']!=epoch for v in self.grad):raise ValueError('Declared gradient epoch incomplete')
   p=out/f'signature_gradients_e{epoch}.pt';torch.save(self.grad,p);saved=torch.load(p,map_location='cpu',weights_only=False);records=[v for v in self.records if v['epoch']==epoch]
   for a,b in zip(saved,records):
    for k,n in enumerate(d.PARAMETERS):
     for key in (*d.LEVELS,'effective'):
      if gradient_stats(a['vectors']['main_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved actual gradient recount differs')
   self.saved.append(epoch);self.grad=[];d.write(out/'signature_gradient_records.json',dict(status='VERIFIED',epochs=self.saved,batch_records=self.records,no_extra_head=True,U_truth_read=False,saved_gradient_recount='VERIFIED',scope='actualmainL128CE versus scheduled separateL90aux; not samepacket or cumulative AdamW contribution'))
 def finish(self):
  if self.steps!=11100 or self.extra!=2750 or self.U_steps!=3774 or self.U_packets!=963900 or self.saved!=list(d.DIAGNOSTIC_EPOCHS) or len(self.records)!=220:raise ValueError('Native/auxiliary/gradient budget incomplete')
  value=dict(status='VERIFIED',steps=self.steps,extra_batches=self.extra,extra_source_packet_forwards=self.extra*90,U_steps=self.U_steps,U_packets=self.U_packets,U_accepted=self.U_accept,gradient_epochs=self.saved,gradient_batches=220,layer_gradient_records=660,head_fit_updates=0,auxiliary_classW_dependency=False,default_head_jointly_trained=True,U_truth_read=False,source_only=True,saved_gradient_recount='VERIFIED')
  d.write(self.out/'signature_training_observation.json',value);return value
