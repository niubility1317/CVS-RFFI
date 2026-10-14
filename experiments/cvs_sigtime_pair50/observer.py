from contextlib import contextmanager
import torch
from . import design as d
from experiments.cvs_neighbor_pair50.sampler import replace
from experiments.cvs_source_layer_order.probe import gradient_stats

def derivative(loss,params):
 if not loss.requires_grad:return [torch.zeros_like(p).flatten().detach().cpu() for p in params]
 gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
 values=[torch.zeros_like(p).flatten().detach().cpu() if g is None else g.detach().flatten().cpu() for p,g in zip(params,gs)]
 if any(not torch.isfinite(v).all() for v in values):raise ValueError('Nonfinite actual SSL gradient')
 return values

class Observer:
 def __init__(self,c):d.validate(c);self.c=c;self.steps=0;self.U_steps=0;self.U_packets=0;self.U_accepted=0;self.records=[];self.raw=[];self.saved=[];self.sampling=[]
 def initialize(self,model,c,out):
  d.validate(c)
  if c!=self.c or model._source_neighbor_sampling['mode']!=c['source_neighbor_sampling']['mode']:raise ValueError('FixedSIG sampler/model context differs')
  self.model=model;self.out=out;self.params=[dict(model.named_parameters())[n] for n in d.PARAMETERS]
  if not all(p.requires_grad for p in self.params) or not model.id_backbone.encoder.core.id_backbone.cls_head.weight.requires_grad:raise ValueError('Default feature/head joint training differs')
 @contextmanager
 def native_scope(self,n):
  if n._trusted_pseudo_loss_hook is not None or n._source_feature_auxiliary_hook is not None:raise ValueError('Another SSL/auxiliary owner active')
  original=n._build_ssdg_wisig_data
  def data(*args,**kw):
   ctx=original(*args,**kw);ctx['unlabeled_loader']=replace(ctx['unlabeled_loader'],self.c['source_neighbor_sampling']['mode']);self.ctx=ctx;return ctx
  n._build_ssdg_wisig_data=data;n._trusted_pseudo_loss_hook=self.observe
  try:yield
  finally:n._build_ssdg_wisig_data=original;n._trusted_pseudo_loss_hook=None
 def observe(self,*,model,args,epoch,batch_idx,loss_l,loss_u,loss_ent,active,logits,pseudo,mask):
  step=(epoch-1)*222+batch_idx
  if model is not self.model or step!=self.steps+1 or bool(active)!=(epoch>=34) or args.lambda_u!=.16 or args.lambda_ent!=.01:raise ValueError('Actual old gate/EMA loss recipe differs')
  self.steps=step
  if active:self.U_steps+=1;self.U_packets+=len(pseudo);self.U_accepted+=int(mask.sum())
  if epoch in d.DIAGNOSTIC_EPOCHS:
   rng=torch.get_rng_state().clone();before=[p.grad.clone() if p.grad is not None else None for p in self.model.parameters()];vectors=dict(main_L_CE=derivative(loss_l,self.params),weighted_U_CE=derivative(.16*loss_u,self.params),weighted_U_CE_entropy=derivative(.16*loss_u+.01*loss_ent,self.params));self.raw.append(dict(step=step,epoch=epoch,vectors=vectors));layers=[]
   for k,name in enumerate(d.PARAMETERS):layers.append(dict(parameter=name,**{key:gradient_stats(vectors['main_L_CE'][k],vectors[key][k]) for key in ('weighted_U_CE','weighted_U_CE_entropy')}))
   if not torch.equal(rng,torch.get_rng_state()) or any((old is None)!=(p.grad is None) or old is not None and not torch.equal(old,p.grad) for old,p in zip(before,self.model.parameters())):raise ValueError('Gradient observer modified accumulated gradients/RNG')
   self.records.append(dict(step=step,epoch=epoch,active=bool(active),U_count=len(pseudo) if active else 0,U_accepted=int(mask.sum()) if active else 0,layers=layers))
  return loss_u
 def epoch(self,model,ctx,rows,out,device):
  epoch=int(rows[-1]['epoch']);history=ctx['unlabeled_loader'].sampler.history;expected=max(0,epoch-33)
  if self.steps!=epoch*222 or len(history)!=expected or expected and history[-1]['delivered_count']!=56700:raise ValueError('Actual native steps/U physical exposure differs')
  self.sampling.append(dict(epoch=epoch,sampler=dict(history[-1]) if expected else None,U_IQ_fetched=bool(expected)))
  d.write(out/'source_sampling_epoch_diagnostic.json',dict(records=self.sampling,source_only=True,U_truth_read=False,head_fit_updates=0,extra_model_forwards=0))
  if epoch in d.DIAGNOSTIC_EPOCHS:
   if len(self.raw)!=222 or any(v['epoch']!=epoch for v in self.raw):raise ValueError('Actual SSL gradient epoch incomplete')
   torch.save(self.raw,out/f'ssl_gradients_e{epoch}.pt');saved=torch.load(out/f'ssl_gradients_e{epoch}.pt',map_location='cpu',weights_only=False);reference=[r for r in self.records if r['epoch']==epoch]
   for a,b in zip(saved,reference):
    for k,name in enumerate(d.PARAMETERS):
     for key in ('weighted_U_CE','weighted_U_CE_entropy'):
      if gradient_stats(a['vectors']['main_L_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved actual SSL gradient recount differs')
   self.raw=[];self.saved.append(epoch);d.write(out/'ssl_gradient_records.json',dict(status='VERIFIED',epochs=self.saved,batch_records=self.records,saved_gradient_recount='VERIFIED',extra_model_forwards=0,head_fit_updates=0,U_truth_read=False,scope='Actual mainL128 vs U256, vectors derivative only, no cumulativeAdamW claim'))
 def finish(self):
  if self.steps!=11100 or self.U_steps!=3774 or self.U_packets!=963900 or self.saved!=list(d.DIAGNOSTIC_EPOCHS) or len(self.records)!=666 or len(self.sampling)!=50:raise ValueError('Complete fixedSIG SSL training budget differs')
  v=dict(status='VERIFIED',steps=11100,U_steps=self.U_steps,U_packets=self.U_packets,U_accepted=self.U_accepted,mode=self.c['source_neighbor_sampling']['mode'],sampler_epochs=17,sampler_packets=963900,gradient_epochs=self.saved,gradient_batches=666,layer_gradient_records=1998,head_fit_updates=0,U_truth_read=False,loss_returned_unchanged=True,extra_model_forwards=0,saved_gradient_recount='VERIFIED')
  d.write(self.out/'ssl_training_observation.json',v);return v
