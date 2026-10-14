from contextlib import contextmanager
import gzip,json,torch
from . import design as d,features
from .objective import recomposition
from experiments.cvs_time_factorial50.observe import Observer as Telemetry
from experiments.cvs_evidence_relation.observe import gradient_summary
from experiments.cvs_u_schedule50.counterfactual import simulate
from experiments.cvs_u_schedule50.observe import label

def aux_names(value):
 if isinstance(value,dict):return {dict(U_increment_norm='aux_increment_norm',U_to_total_step_ratio='aux_to_total_step_ratio').get(k,k):aux_names(v) for k,v in value.items()}
 if isinstance(value,list):return [aux_names(v) for v in value]
 return value

class Observer(Telemetry):
 def initialize(self,model,c,out):
  super().initialize(model,c,out);self.out=out;self.calls=0;self.L_count=0;self.cohort_count=0;self.active_L_count=0;self.log=None;self.grad_ratios=[];self.probe_records=[];self.pending_update=None;self.probe_aux=None;self.epoch_calls=0;self.epoch_cohort=0
 def auxiliary(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
  if model is not self.model or not model.training or model._feature_l_steps!=optimizer_step or labeled_count!=128 or len(labels)!=128:raise ValueError('PA recomposition only actual complete L_s mainbatch')
  self.L_count+=labeled_count
  if not d.active(optimizer_step):return labeled_output['id_feat_cls'].sum()*0.,{'PA_recomposition/active':0.}
  meta=self.native._meta_from_extra(extra)
  if any(k not in meta for k in ('rx_i','day_i','eq_i','sig_i','base_index')):raise ValueError('Actual L physical metadata incomplete')
  if (labels<0).any() or (labels>=6).any() or labeled_output['id_feat_joint'].shape!=(128,160):raise ValueError('Actual true L labels/deployed features differ')
  a,b,c=features.actual_parts(model,labeled_output);head=features.default_head(model)
  result=recomposition(a,b,c,head.weight,head.scale,labels,meta['rx_i'],meta['day_i'],meta['base_index'],self.c['arm'],alpha=.2,coefficient=.1,role='L_s',seed=self.c['model_seed']+91273+optimizer_step*701)
  loss=result['weighted_loss'];facts=result['summary'];count=facts['counts']['cohort_count'];self.calls+=1;self.cohort_count+=count;self.active_L_count+=128;self.epoch_calls+=1;self.epoch_cohort+=count
  record=dict(step=optimizer_step,call=self.calls,main_L_CE_unchanged=True,U_teacher_unchanged=True,head_fit_updates=0,extra_IQ_forward_packets=0,**facts)
  if optimizer_step in d.PROBE_STEPS:
   named=[(n,p) for n,p in model.named_parameters() if n.startswith('id_backbone.') and p.requires_grad and p is not head.weight];params=[p for _,p in named]
   gL=torch.autograd.grad(identity_loss,params,retain_graph=True,allow_unused=True);gA=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);normL=sum(float(g.detach().double().square().sum()) for g in gL if g is not None)**.5;normA=sum(float(g.detach().double().square().sum()) for g in gA if g is not None)**.5
   head_grads=torch.autograd.grad(loss,[head.weight,head.gain,model.id_backbone.encoder.response_gain],retain_graph=True,allow_unused=True)
   if any(g is not None for g in head_grads):raise ValueError('PA auxiliary updates W or gain derivatives')
   ratio=normA/max(normL,1e-12);dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(gL,gA) if x is not None and y is not None)
   audit=dict(aux_to_L_feature_gradient=ratio,aux_L_cosine=dot/(normL*normA) if normL*normA else None,L_norm=normL,aux_norm=normA,groups=dict(L=gradient_summary(named,gL),aux=gradient_summary(named,gA)),W_PA_gain_ref_gain_aux_gradients_None=True)
   record['actual_feature_gradients']=audit;self.grad_ratios.append(ratio);self.probe_aux=dict(step=optimizer_step,loss=loss,audit=audit)
  if self.log is None:self.log=gzip.open(self.out/'PA_recomposition_steps.jsonl.gz','xt',encoding='utf-8')
  self.log.write(json.dumps(record,allow_nan=False)+'\n')
  return loss,{'PA_recomposition/active':1.,'PA_recomposition/loss':loss.detach(),'PA_recomposition/cohort_coverage':count/128.,'PA_recomposition/cross_same_CE_difference':facts['CROSS_RX']['CE']-facts['SAME_RX']['CE'] if count else 0.}
 def probe(self,*,model,optimizer,epoch,batch_idx,phase,optimizer_step,total_loss,unlabeled_loss,labeled_count,unlabeled_count):
  if optimizer_step not in d.PROBE_STEPS:return
  if self.probe_aux is None or self.probe_aux['step']!=optimizer_step or self.pending_update is not None:raise ValueError('PA actual optimizer measurement not aligned with L objective')
  params=[p for group in optimizer.param_groups for p in group['params']];names={id(p):name for name,p in model.named_parameters()};tags=[label(names[id(p)]) for p in params]
  total=torch.autograd.grad(total_loss,params,retain_graph=True,allow_unused=True);base=torch.autograd.grad(total_loss-self.probe_aux['loss'],params,retain_graph=True,allow_unused=True)
  cf=simulate(optimizer,base,total,names=tags,max_grad_norm=5.,step_fn=self.original_step)
  self.pending_update=dict(step=optimizer_step,params=params,before=[p.detach().clone() for p in params],cf=cf,tags=tags,lr=[float(g['lr']) for g in optimizer.param_groups],audit=self.probe_aux['audit']);self.probe_aux=None
 @contextmanager
 def native_scope(self,n):
  self.native=n;builder=n._build_ssdg_wisig_data;core=n.compute_core_losses;step=torch.optim.AdamW.step;self.original_step=step
  def data(*args,**kwargs):
   ctx=builder(*args,**kwargs)
   from .sampler import replace
   ctx['unlabeled_loader']=replace(ctx['unlabeled_loader'],'PAIRS');self.sampler=ctx['unlabeled_loader'].sampler;return ctx
  def losses(*args,**kwargs):
   result=core(*args,**kwargs)
   # Only after native deepcopy created the original ownEMA. These bound
   # training callbacks are never copied to a teacher or prediction model.
   if not hasattr(self.model,'_source_labeled_feature_hook'):
    self.model._source_labeled_feature_hook=self.auxiliary;self.model._source_optimizer_update_probe_hook=self.probe
   return result
  def actual_step(optimizer,*args,**kwargs):
   pending=self.pending_update
   if pending and {id(p) for p in pending['params']}!={id(p) for g in optimizer.param_groups for p in g['params']}:pending=None
   result=step(optimizer,*args,**kwargs)
   if pending:
    error=max(float((p.detach()-q).abs().max()) for p,q in zip(pending['params'],pending['cf']['total_after']));scale=max(1.,max(float(p.detach().abs().max()) for p in pending['params']))
    if error>1e-6*scale:raise ValueError('PA combined shadow differs from actual AdamW')
    movement={}
    for p,old,tag in zip(pending['params'],pending['before'],pending['tags']):
     v=movement.setdefault(tag,dict(changed_values=0,square=0.));v['changed_values']+=int((p.detach()!=old).sum());v['square']+=float((p.detach().double()-old.double()).square().sum())
    record=dict(step=pending['step'],actual_optimizer_steps=1,temporary_steps=2,actual_shadow_max_abs_error=error,actual_shadow_tolerance=1e-6*scale,increment_scope='Actual total loss minus PA auxiliary; original L/U/entropy retained in base',lr=pending['lr'],actual_parameter_updates={k:dict(changed_values=v['changed_values'],norm=v['square']**.5) for k,v in movement.items()},counterfactual=aux_names(pending['cf']['summary']),head_fit_updates=0)
    with (self.out/'actual_PA_optimizer_increment.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
    self.probe_records.append(pending['step']);self.pending_update=None
   return result
  n._build_ssdg_wisig_data=data;n.compute_core_losses=losses;torch.optim.AdamW.step=actual_step
  try:
   with super().native_scope(n):yield
  finally:
   n._build_ssdg_wisig_data=builder;n.compute_core_losses=core;torch.optim.AdamW.step=step
   for name in ('_source_labeled_feature_hook','_source_optimizer_update_probe_hook'):
    if hasattr(getattr(self,'model',None),name):delattr(self.model,name)
   if self.log is not None:self.log.close()
 def epoch(self,model,ctx,rows,out,device):
  epoch=len(rows)
  if self.pending is not None or self.pending_update is not None or self.probe_aux is not None:raise ValueError('Unconsumed PA/U/optimizer observation')
  if self.L_count!=epoch*222*128:raise ValueError('Actual L exposure differs')
  U_count=sum(g['total'] for g in self.groups.values())
  if U_count!=(56700 if epoch>=34 else 0):raise ValueError('Original complete U stage differs')
  with (out/'source_mechanism_epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,U_count=U_count,U_groups=self.groups,L_recomposition_calls=self.epoch_calls,cohort_count=self.epoch_cohort,U_truth_read=False,head_fit_updates=0))+'\n')
  self.groups={};self.epoch_calls=0;self.epoch_cohort=0
  if self.log is not None:self.log.flush()
  d.write(out/'PA_execution.json',dict(epoch=epoch,auxiliary_calls=self.calls,L_packet_observations=self.L_count,active_L_packet_observations=self.active_L_count,common_cohort_packet_observations=self.cohort_count,mean_common_cohort_coverage=self.cohort_count/self.active_L_count if self.active_L_count else None,actual_aux_L_feature_gradient_ratios=self.grad_ratios,optimizer_probe_steps=self.probe_records,U_sampling=self.sampler.history,extra_IQ_forward_packets=0,extra_actual_optimizer_steps=0,head_fit_updates=0,U_truth_read=False,target_access=False,teacher_modified=False,old_L_CE_unchanged=True,default_head_joint_training=True))
