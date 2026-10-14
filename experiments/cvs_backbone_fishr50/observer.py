import gzip,json
from contextlib import contextmanager
import torch
from . import design as d
from .objective import calculate,validate_layout,variance_terms
from experiments.cvs_local_order50.observer import deterministic,derivative
from experiments.cvs_source_layer_order.probe import cohorts,gradient_stats

class Observer:
 def initialize(self,model,c,out):
  from experiments.cvs_nocfo_foundation.source import load_source
  from experiments.cvs_nocfo_foundation import design as nd
  d.validate(c);self.c=c;self.model=model;self.out=out;self.steps=self.extra=self.U_steps=self.U_packets=self.U_accept=0;self.records=[];self.saved=[];self.grad=[]
  L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V;self.L=L;old=cohorts(L);self.ix=[torch.cat((old[2*j],old[2*j+1])) for j in range(35)]
  for ix in self.ix:validate_layout(*(L[k][ix] for k in ('y','rx','day')),[L['ids'][i] for i in ix.tolist()])
  self.params=[dict(model.named_parameters())[n] for n in d.PARAMETERS];self.head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
  if not all(p.requires_grad for p in self.params+[self.head]):raise ValueError('Default feature/head joint training disabled')
  self.stream=gzip.open(out/'backbone_fishr_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3)
 @contextmanager
 def native_scope(self,n):
  if n._source_feature_auxiliary_hook is not None or n._trusted_pseudo_loss_hook is not None:raise ValueError('Another auxiliary owner active')
  n._source_feature_auxiliary_hook=self.aux;n._trusted_pseudo_loss_hook=self.base
  try:yield
  finally:n._source_feature_auxiliary_hook=None;n._trusted_pseudo_loss_hook=None;self.stream.close()
 def base(self,*,model,args,epoch,batch_idx,loss_l,loss_u,loss_ent,active,logits,pseudo,mask):
  step=(epoch-1)*222+batch_idx
  if model is not self.model or step!=self.steps+1 or bool(active)!=(epoch>=34) or (args.lambda_u,args.lambda_ent,args.label_smoothing)!=(.16,.01,.01):raise ValueError('Original source/U/default CE recipe differs')
  self.steps=step;self.main_CE=loss_l
  if active:self.U_steps+=1;self.U_packets+=len(pseudo);self.U_accept+=int(mask.sum())
  return loss_u
 def aux(self,model,step):
  if model is not self.model or step!=self.steps:raise ValueError('Source auxiliary step differs')
  epoch=(step-1)//222+1;batch=(step-1)%222+1
  if batch%d.INTERVAL:return self.main_CE*0,dict(backbone_fishr_extra_forward=0.)
  rank=self.extra%d.COHORTS;ix=self.ix[rank];self.extra+=1;dev=self.params[0].device;y,rx,day=(self.L[k][ix].to(dev) for k in ('y','rx','day'));ids=[self.L['ids'][i] for i in ix.tolist()]
  with deterministic(model):losses,detail,matrices=calculate(model,self.L['x'][ix].to(dev),y,rx,day,ids)
  mode=self.c['backbone_fishr']['mode'];ramp=min(1.,max(0.,(epoch-5)/10));weight=0. if mode=='ZERO' else d.COEFFICIENT*ramp;selected=losses['RAW' if mode=='ZERO' else mode];effective=weight*selected
  if any(torch.autograd.grad(loss,self.head,allow_unused=True,retain_graph=True)[0] is not None for loss in losses.values()):raise ValueError('Regularizer updates classW')
  if not torch.isfinite(effective):raise FloatingPointError('Nonfinite backbone variance regularizer')
  record=dict(step=step,epoch=epoch,batch=batch,cohort_rank=rank,mode=mode,ramp=ramp,coefficient=weight,raw_losses={k:float(v.detach()) for k,v in losses.items()},effective_loss=float(effective.detach()),source_only=True,extra_source_packets=180,detail=detail)
  if epoch in d.DIAGNOSTIC_EPOCHS:
   vectors=dict(main_CE=derivative(self.main_CE,self.params),effective=derivative(effective,self.params));vectors.update({key:derivative(d.COEFFICIENT*loss,self.params) for key,loss in losses.items()})
   self.grad.append(dict(step=step,epoch=epoch,vectors=vectors,parameter_gradient_matrices=[g.detach().cpu() for g in matrices],y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids))
   self.records.append(dict(step=step,epoch=epoch,raw_losses=record['raw_losses'],layers=[dict(parameter=name,**{key:gradient_stats(vectors['main_CE'][k],vectors[key][k]) for key in (*d.LEVELS,'effective')}) for k,name in enumerate(d.PARAMETERS)]))
  self.stream.write(json.dumps(record,allow_nan=False)+chr(10));self.main_CE=None
  return effective,dict(backbone_fishr_extra_forward=1.,backbone_fishr_raw=record['raw_losses']['RAW'],backbone_fishr_centered=record['raw_losses']['TX_DAY_CENTERED'],backbone_fishr_effective=record['effective_loss'])
 def epoch(self,model,ctx,rows,out,device):
  epoch=int(rows[-1]['epoch']);self.stream.flush()
  if self.steps!=epoch*222 or self.extra!=epoch*d.EXTRA_PER_EPOCH:raise ValueError('Actual native/extra exposure incomplete')
  if epoch in d.DIAGNOSTIC_EPOCHS:
   if len(self.grad)!=d.EXTRA_PER_EPOCH or any(v['epoch']!=epoch for v in self.grad):raise ValueError('Actual gradient epoch incomplete')
   path=out/f'backbone_fishr_gradients_e{epoch}.pt';torch.save(self.grad,path);saved=torch.load(path,map_location='cpu',weights_only=False);records=[v for v in self.records if v['epoch']==epoch]
   for a,b in zip(saved,records):
    recomputed={key:[] for key in d.LEVELS}
    for G in a['parameter_gradient_matrices']:
     losses,_=variance_terms(G,a['y'],a['rx'],a['day'])
     for key in d.LEVELS:recomputed[key].append(float(losses[key]))
    for key in d.LEVELS:
     value=sum(recomputed[key])/2
     if abs(value-b['raw_losses'][key])>1e-8*max(1.,abs(value)):raise ValueError('Saved per-packet-gradient variance recount differs')
    for k,name in enumerate(d.PARAMETERS):
     for key in (*d.LEVELS,'effective'):
      if gradient_stats(a['vectors']['main_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved effective gradient recount differs')
   self.saved.append(epoch);self.grad=[];d.write(out/'backbone_fishr_gradient_records.json',dict(status='VERIFIED',epochs=self.saved,batch_records=self.records,no_extra_head=True,U_truth_read=False,saved_gradient_recount='VERIFIED',scope='Actual separateL180 backbone-gradient objective vs mainL128CE; no cumulativeAdamW claim'))
 def finish(self):
  if self.steps!=11100 or self.extra!=d.EXTRA_BATCHES or self.U_steps!=3774 or self.U_packets!=963900 or self.saved!=list(d.DIAGNOSTIC_EPOCHS) or len(self.records)!=52:raise ValueError('Native/backbone variance/gradient budget incomplete')
  value=dict(status='VERIFIED',steps=self.steps,extra_batches=self.extra,extra_source_packet_forwards=self.extra*180,U_steps=self.U_steps,U_packets=self.U_packets,U_accepted=self.U_accept,gradient_epochs=self.saved,gradient_batches=52,layer_gradient_records=156,actual_per_sample_matrices=104,head_fit_updates=0,auxiliary_classW_dependency=False,auxiliary_uses_current_default_W_as_constant=True,default_head_jointly_trained=True,U_truth_read=False,source_only=True,saved_gradient_recount='VERIFIED')
  d.write(self.out/'backbone_fishr_training_observation.json',value);return value
