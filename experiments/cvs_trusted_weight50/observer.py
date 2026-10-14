"""Actual pre-backward native L/U losses; no extra forward/RNG or head fitting."""
import gzip,json
from contextlib import contextmanager
import torch
from . import design as d
from experiments.cvs_source_weight.weights import losses
from experiments.cvs_source_weight.design import PARAMETERS

def records(vectors):
 parts=[torch.split(v,[3456,6912,9216]) for v in vectors];out=[]
 for name,l,u,w,e in zip(PARAMETERS,*parts):
  l,u,w,e=(x.double() for x in (l,u,w,e));nl,nu,nw,ne=(float(x.norm()) for x in (l,u,w,e));au,aw=u+e,w+e
  if any(not torch.isfinite(x).all() for x in (l,u,w,e)):raise ValueError('Nonfinite actual feature gradients')
  cosine=lambda a,b:float(torch.dot(a,b)/(a.norm()*b.norm())) if float(a.norm()*b.norm())>1e-12 else None
  out.append(dict(parameter=name,elements=len(l),L_CE_norm=nl,lambdaU_uniform_norm=nu,lambdaU_weighted_norm=nw,lambdaEnt_norm=ne,uniform_U_to_L=nu/nl if nl>1e-12 else None,weighted_U_to_L=nw/nl if nl>1e-12 else None,uniform_totalU_to_L=float(au.norm())/nl if nl>1e-12 else None,weighted_totalU_to_L=float(aw.norm())/nl if nl>1e-12 else None,weighted_uniform_cosine=cosine(w,u),uniformU_L_cosine=cosine(au,l),weightedU_L_cosine=cosine(aw,l)))
 return out

class Observer:
 def initialize(self,model,c,out):
  d.validate(c);self.c=c;self.out=out;self.steps=0;self.U_steps=0;self.U_packets=0;self.U_accepted=0;self.grad=[];self.grad_records=[];self.epochs_saved=[];self.model=model
  named=dict(model.named_parameters());self.params=[named[n] for n in PARAMETERS]
  head=model.id_backbone.encoder.core.id_backbone.cls_head
  if [p.numel() for p in self.params]!=[3456,6912,9216] or not all(p.requires_grad for p in self.params) or not head.weight.requires_grad or head.weight.shape!=(6,160) or head.scale!=30:raise ValueError('Default joint classifier/feature gradient objects differ')
  self.stream=gzip.open(out/'trusted_weight_steps.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
 @contextmanager
 def native_scope(self,native):
  old=native._trusted_pseudo_loss_hook
  if old is not None:raise ValueError('Another trusted loss owner is active')
  native._trusted_pseudo_loss_hook=self.loss
  try:yield
  finally:
   native._trusted_pseudo_loss_hook=old
   if hasattr(self,'stream'):self.stream.close()
 def loss(self,*,model,args,epoch,batch_idx,loss_l,loss_u,loss_ent,active,logits,pseudo,mask):
  expected=(epoch-1)*222+batch_idx
  if model is not self.model or expected!=self.steps+1 or not 1<=batch_idx<=222 or bool(active)!=(epoch>=34) or args.lambda_u!=.16 or not model.id_backbone.encoder.core.id_backbone.cls_head.weight.requires_grad:raise ValueError('Actual step/stage/default joint head differs')
  self.steps+=1;uniform=loss_u;weighted=loss_u;step=dict(step=self.steps,epoch=epoch,batch=batch_idx,active=bool(active),lambda_u=args.lambda_u,lambda_ent=args.lambda_ent,weighted_training=self.c['trusted_weight']['active'],accepted=0,U_count=0)
  if active:
   old_mask=mask.clone();v=losses(logits,pseudo,mask)
   if not torch.equal(old_mask,mask) or not torch.equal(mask,v['accept']) or v['raw_weight'].requires_grad or not torch.allclose(v['uniform'],uniform.detach(),atol=2e-6,rtol=2e-6):raise ValueError('Native gate or uniform accepted CE differs')
   weighted=v['weighted'];self.U_steps+=1;self.U_packets+=len(pseudo);accepted=int(mask.sum());self.U_accepted+=accepted
   step.update(accepted=accepted,U_count=len(pseudo),raw_min=float(v['raw_weight'][mask].min()) if accepted else None,raw_max=float(v['raw_weight'][mask].max()) if accepted else None,accepted_h_mean=float(v['hardness'][mask].mean()) if accepted else None,uniform_CE=float(uniform.detach()),weighted_CE=float(weighted.detach()))
  if epoch in (33,34,50):
   grad=lambda loss:torch.cat([g.detach().flatten().cpu() for g in torch.autograd.grad(loss,self.params,retain_graph=True)])
   vectors=(grad(loss_l),grad(args.lambda_u*uniform),grad(args.lambda_u*weighted),grad(args.lambda_ent*loss_ent));r=records(vectors);self.grad.append(dict(step=self.steps,epoch=epoch,batch=batch_idx,active=bool(active),accepted=step['accepted'],vectors=vectors));self.grad_records.append(dict(step=self.steps,epoch=epoch,batch=batch_idx,active=bool(active),accepted=step['accepted'],records=r))
  result=weighted if self.c['trusted_weight']['active'] else uniform
  if not torch.isfinite(result):raise ValueError('Nonfinite trusted loss')
  self.stream.write(json.dumps(step,allow_nan=False)+'\n');return result
 def epoch(self,model,ctx,rows,out,device):
  epoch=rows[-1]['epoch']
  if self.steps!=epoch*222:raise ValueError('Native epoch step observation incomplete')
  self.stream.flush()
  if epoch in (33,34,50):
   if len(self.grad)!=222 or any(x['epoch']!=epoch for x in self.grad):raise ValueError('Actual declared epoch gradients incomplete')
   path=out/f'trusted_gradients_e{epoch}.pt';torch.save(self.grad,path);saved=torch.load(path,map_location='cpu',weights_only=False)
   prior=[x for x in self.grad_records if x['epoch']==epoch]
   if len(saved)!=222 or any(records(x['vectors'])!=y['records'] or x['step']!=y['step'] for x,y in zip(saved,prior)):raise ValueError('Saved actual gradient recount differs')
   self.epochs_saved.append(epoch);self.grad=[]
   d.write(out/'trusted_gradient_records.json',dict(status='VERIFIED',epochs=list(self.epochs_saved),batch_records=self.grad_records,head_fit=False,U_truth_read=False,extra_forward=0,extra_RNG_draws=0,source_only=True,saved_gradient_recount='VERIFIED'))
 def finish(self):
  if self.steps!=11100 or self.U_steps!=3774 or self.U_packets!=963900 or self.epochs_saved!=[33,34,50] or len(self.grad_records)!=666 or any(len(x['records'])!=3 for x in self.grad_records):raise ValueError('Trusted weighting/actual derivative budget incomplete')
  v=dict(status='VERIFIED',steps=self.steps,U_steps=self.U_steps,U_packets=self.U_packets,U_accepted=self.U_accepted,epochs=self.epochs_saved,batch_records=666,layer_gradient_records=1998,weight_active=self.c['trusted_weight']['active'],raw_bounds=[.75,1.25],old_gate_unchanged=True,head_fit=False,default_head_jointly_trained=True,extra_forward=0,extra_RNG_draws=0,U_truth_read=False,source_only=True,saved_gradient_recount='VERIFIED')
  d.write(self.out/'trusted_training_observation.json',v);return v
