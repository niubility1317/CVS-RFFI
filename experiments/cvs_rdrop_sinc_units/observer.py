"""Actual optimizer-step witnesses for RAW, original CENTER and unified policies."""
import gzip,json
from contextlib import contextmanager
import torch
from . import design as d
from .optimizer import native_scope
from experiments.cvs_sigtime_rdrop_legacy.source import Observer as Original

def summarize_snapshots(rows):
 return [dict(step=r['step'],epoch=r['epoch'],parameters=[dict(parameter=n,actual_leaf_gradient_norm=float(v['gradient'].double().norm()),changed_elements=int((v['physical_after']!=v['physical_before']).sum()),delta_norm_Hz=float((v['physical_after'].double()-v['physical_before'].double()).norm()),maximum_abs_delta_Hz=float((v['physical_after']-v['physical_before']).abs().max()),optimizer_step=int(v['optimizer_step']),finite=all(bool(torch.isfinite(v[k]).all()) for k in ('before','after','physical_before','physical_after','gradient','first_moment','second_moment'))) for n,v in r['parameters'].items()]) for r in rows]
class Observer(Original):
 def __init__(self,c):
  d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def initialize(self,m,c,out):super().initialize(m,c,out);self.model=m
 @contextmanager
 def native_scope(self,native):
  with native_scope(self,self.c['sinc_optimizer_units']['mode']):yield
 def optimizer_created(self,opt):
  def pre(opt,args,kwargs):
   step=opt._sinc_unit_steps+1;ep=(step-1)//222+1;batch=(step-1)%222+1
   if ep in d.SINC_DIAGNOSTIC_EPOCHS and batch%16==0:
    if self.pending is not None:raise ValueError('Overlapping actual Sinc step')
    pol=opt._sinc_unit_policy;values={}
    for n,p,o in zip(pol['names'],pol['sinc'],pol['origins']):
     if p.grad is None:raise ValueError('Missing actual total Sinc derivative')
     before=opt._sinc_pre_step[n];physical=before if o is None else o+pol['scale']*before
     values[n]=dict(before=before.clone(),physical_before=physical.detach().clone(),gradient=p.grad.detach().clone())
    self.pending=dict(step=step,epoch=ep,parameters=values)
  def post(opt,args,kwargs):
   if self.pending is None:return
   record=self.pending
   if record['step']!=opt._sinc_unit_steps:raise ValueError('Actual optimizer ownership differs')
   pol=opt._sinc_unit_policy
   for n,p,o in zip(pol['names'],pol['sinc'],pol['origins']):
    v=record['parameters'][n];s=opt.state[p];physical=p if o is None else o+pol['scale']*p
    v.update(after=p.detach().clone(),physical_after=physical.detach().clone(),first_moment=s['exp_avg'].detach().clone(),second_moment=s['exp_avg_sq'].detach().clone(),optimizer_step=int(s['step']))
    for key in tuple(v):
     if torch.is_tensor(v[key]):v[key]=v[key].cpu()
   self.raw.append(record);self.pending=None
  opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);ep=int(rows[-1]['epoch']);q=d.read(self.out/'source_training_mechanism.json');q['sinc_coordinates']=self.c['combination']['Sinc_coordinates'];q['A_F_arm']=self.c['arm'];d.write(self.out/'source_training_mechanism.json',q)
  if self.optimizer._sinc_unit_steps!=ep*222 or len(self.optimizer._sinc_unit_lr_records)!=ep*222:raise ValueError('Actual optimizer/RDrop budgets differ')
  if ep in d.SINC_DIAGNOSTIC_EPOCHS:
   if len(self.raw)!=13 or any(r['epoch']!=ep for r in self.raw):raise ValueError('Missing actual Sinc snapshots')
   p=out/f'sinc_optimizer_gradients_updates_e{ep}.pt';torch.save(self.raw,p);readback=torch.load(p,map_location='cpu',weights_only=False)
   if summarize_snapshots(readback)!=summarize_snapshots(self.raw):raise ValueError('Saved actual update differs')
   self.records.extend(summarize_snapshots(readback));self.saved.append(ep);self.raw=[]
   d.write(out/'sinc_optimizer_update_records.json',dict(status='VERIFIED',epochs=self.saved,records=self.records,gradient_definition='Actual total RDrop+oldU leaf derivative after registered globalclip; physical_before/after include complete decay+adaptive step',head_fit_updates=0,extra_model_forwards=0))
 def finish(self):
  if self.pending is not None or self.saved!=list(d.SINC_DIAGNOSTIC_EPOCHS) or len(self.records)!=52 or self.optimizer._sinc_unit_steps!=11100:raise ValueError('Incomplete actual Sinc witness')
  clips=self.optimizer._sinc_physical_clip_calls
  if self.c['sinc_optimizer_units']['mode']=='CENTER_UNITS' and clips!=11100:raise ValueError('Physical globalclip budget incomplete')
  with gzip.open(self.out/'sinc_optimizer_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3) as stream:
   for r in self.optimizer._sinc_unit_lr_records:stream.write(json.dumps(r,allow_nan=False)+'\n')
  value=dict(status='VERIFIED',steps=11100,mode=self.c['sinc_optimizer_units']['mode'],Sinc_parameters=48,sample_rate_Hz=25000000.,physical_clip_calls=clips,diagnostic_batches=52,Sinc_tensor_records=104,extra_model_forwards_vs_RDrop=0,extra_optimizer_updates=0,default_head_jointly_trained=True,head_fit_updates=0,U_truth_read=False,target_access=False)
  d.write(self.out/'sinc_optimizer_observation.json',value);return value
