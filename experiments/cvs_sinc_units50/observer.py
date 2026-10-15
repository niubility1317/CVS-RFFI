import gzip,json
from contextlib import contextmanager
import torch
from . import design as d
from .optimizer import native_scope as optimizer_scope,NAMES
from experiments.cvs_sigtime_pair50.observer import Observer as Parent

def summarize_snapshots(rows):
 result=[]
 for r in rows:
  result.append(dict(step=r['step'],epoch=r['epoch'],parameters=[dict(parameter=name,actual_gradient_norm=float(v['gradient'].double().norm()),changed_elements=int((v['after']!=v['before']).sum()),delta_norm_Hz=float((v['after'].double()-v['before'].double()).norm()),maximum_abs_delta_Hz=float((v['after']-v['before']).abs().max()),optimizer_step=int(v['optimizer_step']),finite=all(bool(torch.isfinite(v[k]).all()) for k in ('before','after','gradient','first_moment','second_moment'))) for name,v in r['parameters'].items()]))
 return result

class Observer(Parent):
 def __init__(self,c):
  d.validate(c);self.c=c;self.steps=0;self.U_steps=0;self.U_packets=0;self.U_accepted=0;self.records=[];self.raw=[];self.saved=[];self.sampling=[];self.sinc_raw=[];self.sinc_saved=[];self.sinc_records=[];self.pending_sinc=None;self.optimizer=None
 def initialize(self,model,c,out):
  d.validate(c)
  if c!=self.c or model._source_neighbor_sampling['mode']!='RANDOM':raise ValueError('Fixed native random U context differs')
  self.model=model;self.out=out;self.params=[dict(model.named_parameters())[n] for n in d.PARAMETERS]
  if not all(p.requires_grad for p in self.params) or not model.id_backbone.encoder.core.id_backbone.cls_head.weight.requires_grad:raise ValueError('Original joint body/head disabled')
 @contextmanager
 def native_scope(self,n):
  with super().native_scope(n),optimizer_scope(self,self.c['sinc_optimizer_units']['mode']):yield
 def optimizer_created(self,opt):
  def pre(opt,args,kwargs):
   step=opt._sinc_unit_steps+1;epoch=(step-1)//222+1;batch=(step-1)%222+1
   if epoch in d.SINC_DIAGNOSTIC_EPOCHS and batch%16==0:
    if self.pending_sinc is not None:raise ValueError('Overlapping Sinc step observation')
    named=dict(self.model.named_parameters());p={}
    for name in NAMES:
     parameter=named[name]
     if parameter.grad is None:raise ValueError('Actual Sinc gradient missing')
     p[name]=dict(before=parameter.detach().clone(),gradient=parameter.grad.detach().clone())
    self.pending_sinc=dict(step=step,epoch=epoch,parameters=p)
  def post(opt,args,kwargs):
   if self.pending_sinc is None:return
   r=self.pending_sinc
   if r['step']!=opt._sinc_unit_steps:raise ValueError('Sinc step owner differs')
   named=dict(self.model.named_parameters())
   for name,v in r['parameters'].items():
    p=named[name];state=opt.state[p];v.update(after=p.detach().clone(),first_moment=state['exp_avg'].detach().clone(),second_moment=state['exp_avg_sq'].detach().clone(),optimizer_step=int(state['step']))
    for key in ('before','after','gradient','first_moment','second_moment'):v[key]=v[key].cpu()
   self.sinc_raw.append(r);self.pending_sinc=None
  opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
 def epoch(self,model,ctx,rows,out,device):
  super().epoch(model,ctx,rows,out,device);epoch=int(rows[-1]['epoch'])
  if self.optimizer._sinc_unit_steps!=epoch*222 or len(self.optimizer._sinc_unit_lr_records)!=epoch*222:raise ValueError('Actual unit optimizer update count differs')
  if epoch in d.SINC_DIAGNOSTIC_EPOCHS:
   if len(self.sinc_raw)!=13 or any(r['epoch']!=epoch for r in self.sinc_raw):raise ValueError('Fixed Sinc diagnostic exposure differs')
   path=out/f'sinc_optimizer_gradients_updates_e{epoch}.pt';torch.save(self.sinc_raw,path);saved=torch.load(path,map_location='cpu',weights_only=False);summary=summarize_snapshots(saved)
   if summary!=summarize_snapshots(self.sinc_raw):raise ValueError('Saved Sinc update recount differs')
   self.sinc_records.extend(summary);self.sinc_saved.append(epoch);self.sinc_raw=[]
   d.write(out/'sinc_optimizer_update_records.json',dict(status='VERIFIED',epochs=self.sinc_saved,records=self.sinc_records,gradient_definition='Actual total native L+oldU/entropy gradients after unchangedglobalclip, beforeAdamW; not perobjective attribution',extra_model_forwards=0,head_fit_updates=0))
 def finish(self):
  parent=super().finish()
  if self.pending_sinc is not None or self.sinc_saved!=list(d.SINC_DIAGNOSTIC_EPOCHS) or len(self.sinc_records)!=52 or self.optimizer._sinc_unit_steps!=11100:raise ValueError('Complete actual Sinc learning witness differs')
  with gzip.open(self.out/'sinc_optimizer_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3) as stream:
   for r in self.optimizer._sinc_unit_lr_records:stream.write(json.dumps(r,allow_nan=False)+chr(10))
  value=dict(status='VERIFIED',steps=11100,mode=self.c['sinc_optimizer_units']['mode'],sample_rate_Hz=25000000.,Sinc_parameters=48,diagnostic_epochs=self.sinc_saved,diagnostic_batches=52,Sinc_tensor_records=104,extra_model_forwards=0,extra_optimizer_updates=0,original_U_observation=parent,default_head_jointly_trained=True,head_fit_updates=0,U_truth_read=False,scope='Actual original total clipped gradients and numerical Sinc updates, not an independent head fit or optimizer contribution decomposition')
  d.write(self.out/'sinc_optimizer_observation.json',value);return value
