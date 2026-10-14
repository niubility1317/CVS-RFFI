from contextlib import contextmanager
import gzip,json,torch
from . import design as d
from experiments.cvs_time_factorial50.observe import Observer as Telemetry

def label(name):
 for group in ('t1','t2','t3','pa_b1','pa_b2','pa_b3','f_proj','t_proj','pa_proj','tf_fuse','response_projection','sinc'):
  if '.'+group+'.' in name:return group
 if name.endswith('.response_gain'):return 'ref_gain'
 if name.endswith('.cls_head.gain'):return 'PA_gain'
 if name.endswith('.cls_head.weight'):return 'default_class_W'
 return 'other'

class Observer(Telemetry):
 def initialize(self,model,c,out):
  super().initialize(model,c,out);self.out=out;self.calendar=c['calendar']['unlabeled_epochs'];self.pending_update=None;self.L_count=0;self.U_count=0;self.step_count=0;self.proxies=[];self.optimizer_probes=[]
 def epoch_start(self,*,model,teacher,epoch,context,optimizer):
  if epoch not in self.calendar:return
  ordinal=self.calendar.index(epoch)+1
  if ordinal not in d.PROBE_U_ORDINALS:return
  if model._feature_l_steps!=(epoch-1)*222:raise ValueError('Proxy not at current preU epoch stage')
  from .proxy import snapshot
  result=snapshot(model,teacher,context,epoch,self.c,self.proxy_native,self.out);self.proxies.append(dict(epoch=epoch,ordinal=ordinal,teacher_accuracy=result['teacher_accuracy'],teacher_ECE15=result['teacher_ECE15'],accepted=result['records']['overall']['stages']['accepted']))
 def probe(self,*,model,optimizer,epoch,batch_idx,phase,optimizer_step,total_loss,unlabeled_loss,labeled_count,unlabeled_count):
  self.L_count+=int(labeled_count);self.U_count+=int(unlabeled_count);self.step_count+=1
  active=epoch in self.calendar
  expected_U=(256 if batch_idx<222 else 124) if active else 0
  if phase!=('pseudo' if active else 'label') or int(unlabeled_count)!=expected_U:raise ValueError('Actual U calendar/packet count differs')
  if labeled_count!=128 or optimizer_step!=self.step_count or model._feature_l_steps!=optimizer_step:raise ValueError('Actual L/update exposure differs')
  if not active or batch_idx!=1 or self.calendar.index(epoch)+1 not in d.PROBE_U_ORDINALS:return
  if self.pending_update is not None:raise ValueError('Unconsumed optimizer measurement')
  params=[p for g in optimizer.param_groups for p in g['params']];names={id(p):n for n,p in model.named_parameters()};tags=[label(names[id(p)]) for p in params]
  total=torch.autograd.grad(total_loss,params,allow_unused=True,retain_graph=True);base=torch.autograd.grad(total_loss-unlabeled_loss,params,allow_unused=True,retain_graph=True)
  from .counterfactual import simulate
  cf=simulate(optimizer,base,total,names=tags,max_grad_norm=5.,step_fn=self.original_step)
  self.pending_update=dict(optimizer=optimizer,params=params,before=[p.detach().clone() for p in params],counterfactual=cf,tags=tags,step=optimizer_step,epoch=epoch,U_ordinal=self.calendar.index(epoch)+1,lr=[float(g['lr']) for g in optimizer.param_groups])
 def epoch(self,model,ctx,rows,out,device):
  epoch=len(rows)
  if self.pending is not None or self.pending_update is not None:raise ValueError('Unconsumed U/optimizer observation')
  count=sum(v['total'] for v in self.groups.values());expected=56700 if epoch in self.calendar else 0
  if count!=expected:raise ValueError('Complete U epoch exposure differs')
  with (out/'source_mechanism_epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,U_count=count,active=bool(expected),groups=self.groups,U_truth_read=False,gate_changed=False,head_fit_updates=0))+'\n')
  self.groups={};d.write(out/'calendar_execution.json',dict(epoch=epoch,observed_optimizer_steps=self.step_count,L_packet_observations=self.L_count,U_packet_observations=self.U_count,completed_U_epochs=[e for e in self.calendar if e<=epoch],U_sampling=self.sampler.history,current_teacher_proxies=self.proxies,optimizer_probe_steps=self.optimizer_probes,extra_V_forward_packets=len(self.proxies)*54000,head_fit_updates=0,U_truth_read=False,target_access=False,actual_extra_optimizer_steps=0))
 @contextmanager
 def native_scope(self,n):
  data=n._build_ssdg_wisig_data;context=n.compute_core_losses;step=torch.optim.AdamW.step;self.original_step=step
  self.proxy_native={k:getattr(n,k) for k in ('_threshold_mask','_temporal_mask_tensor','_meta_from_extra')}
  def prepare(*args,**kwargs):
   ctx=data(*args,**kwargs)
   from .sampler import replace
   ctx['unlabeled_loader']=replace(ctx['unlabeled_loader'],'PAIRS');self.sampler=ctx['unlabeled_loader'].sampler;return ctx
  def losses(*args,**kwargs):
   result=context(*args,**kwargs)
   # EMA already deepcopied the student before the first labeled loss.
   if not hasattr(self.model,'_source_optimizer_update_probe_hook'):
    self.model._source_optimizer_update_probe_hook=self.probe;self.model._source_epoch_probe_hook=self.epoch_start
   return result
  def actual_step(optimizer,*args,**kwargs):
   pending=self.pending_update;pending=pending if pending is not None and pending['optimizer'] is optimizer else None
   result=step(optimizer,*args,**kwargs)
   if pending:
    expected=pending['counterfactual']['total_after'];params=pending['params'];error=max(float((p.detach()-q).abs().max()) for p,q in zip(params,expected));scale=max(1.,max(float(p.detach().abs().max()) for p in params))
    if error>1e-6*scale:raise ValueError('Combined shadow differs from actual AdamW update')
    entry=dict(step=pending['step'],epoch=pending['epoch'],U_ordinal=pending['U_ordinal'],lr=pending['lr'],counterfactual=pending['counterfactual']['summary'],actual_shadow_max_abs_error=error,actual_shadow_FP32_tolerance=1e-6*scale,actual_optimizer_steps=1,shadow_steps=2,real_optimizer_state_preserved_during_shadows=True,U_scope='.16selectedpseudoCE+.01allUentropy',base_scope='actualtotalloss minus thoseUterms; samemoments,perbranchclip5,weightdecay',head_fit_updates=0)
    with (self.out/'actual_U_optimizer_increment.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
    self.optimizer_probes.append(pending['step']);self.pending_update=None
   return result
  n._build_ssdg_wisig_data=prepare;n.compute_core_losses=losses;torch.optim.AdamW.step=actual_step
  try:
   with super().native_scope(n):yield
  finally:
   n._build_ssdg_wisig_data=data;n.compute_core_losses=context;torch.optim.AdamW.step=step
   for key in ('_source_optimizer_update_probe_hook','_source_epoch_probe_hook'):
    if hasattr(getattr(self,'model',None),key):delattr(self.model,key)
