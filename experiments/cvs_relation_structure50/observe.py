from contextlib import contextmanager
import gzip,json,torch
from . import design as d,components
from .objective import objective,RXS
from .model import classifier
from experiments.cvs_time_factorial50.observe import Observer as Telemetry

def gradient_summary(named,values):
 groups={}
 for (name,p),g in zip(named,values):
  for label in ('t1','t2','t3','pa_b1','pa_b2','pa_b3','t_proj','f_proj','pa_proj','response_projection','tf_fuse'):
   if '.'+label+'.' in name:
    v=groups.setdefault(label,dict(parameters=0,gradient_square=0.,nonzero_values=0));v['parameters']+=p.numel()
    if g is not None:v['gradient_square']+=float(g.detach().double().square().sum());v['nonzero_values']+=int((g.detach()!=0).sum())
 return {k:dict(parameters=v['parameters'],norm=v['gradient_square']**.5,nonzero_values=v['nonzero_values']) for k,v in groups.items()}

class Observer(Telemetry):
 def prepare(self,ctx):
  dataset=ctx['train_loader'].dataset
  if len(dataset)!=6300:raise ValueError('Only exact legal L6300 anchor sampling')
  cells={}
  for i in range(len(dataset)):
   _,y,_,meta=dataset[i];key=(int(meta['rx_i']),int(y),int(meta['day_i']));cells.setdefault(key,[]).append(i)
  if set(cells)!={(r,t,day) for r in RXS for t in range(6) for day in (1,2,3)} or any(len(v)!=70 for v in cells.values()):raise ValueError('Exact source L TX/RX/day cells differ')
  self.anchor_dataset=dataset;self.anchor_cells=cells;self.U_sampler=ctx['unlabeled_loader'].sampler
 def initialize(self,model,c,out):
  super().initialize(model,c,out);self.out=out;self.calls=0;self.U_packets=0;self.refreshes=0;self.bank=None;self.log=None;self.criterion_values=[];self.nonzero_sum=0.;self.pending_update=None
 def refresh(self,teacher,step):
  seed=self.c['model_seed']+93217;gen=torch.Generator().manual_seed(seed);indices=[]
  for r in RXS:
   for tx in range(6):
    for day in (1,2,3):
     pool=self.anchor_cells[r,tx,day];indices.extend(pool[j] for j in torch.randperm(70,generator=gen)[:2].tolist())
  samples=[self.anchor_dataset[i] for i in indices];dev=next(teacher.parameters()).device;x=torch.stack([s[0] for s in samples]).to(dev)
  flags=[(m,m.training) for m in teacher.modules()];devices=[dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type=='cuda' else []
  with torch.random.fork_rng(devices=devices),torch.no_grad():
   teacher.eval();out=teacher.forward_identity_only(x);self.bank=out['id_feat_joint'].detach().reshape(5,6,6,160)
  for m,flag in flags:m.training=flag
  self.refreshes+=1;self.bank_step=step-1
  d.write(self.out/'anchor_refresh.json',dict(refreshes=self.refreshes,last_teacher_successful_update=self.bank_step,count=180,role='L_s',physical_ids=[[int(s[3][k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')] for s in samples],TX_labels_used_for_balanced_L_sampling_only=True,U_truth_read=False,target_access=False))
 def auxiliary(self,*,model,teacher,student_output,teacher_output,extra,domains,optimizer_step,identity_loss,pseudo_loss,entropy_loss):
  step=optimizer_step
  if model is not self.model or step!=model._feature_l_steps or not 7326<step<=11100 or teacher is model:raise ValueError('Original ownEMA/U-stage update contract differs')
  from experiments.cvs_phase1_overlay.model import native_modules
  meta=native_modules()._meta_from_extra(extra)
  if any(k in meta for k in ('tx','tx_i','true_tx_i','tx_label','y_tx')):raise ValueError('U TX metadata forbidden')
  rx=meta['rx_i'].to(student_output['id_feat_joint'].device);sp=components.parts(model,student_output);tp=components.parts(teacher,teacher_output)
  # Capture weakU parts BEFORE anchor refresh overwrites module-local caches.
  if self.bank is None or (step-7327)%222==0:self.refresh(teacher,step)
  total,raw,facts=objective(sp,tp,self.bank,rx,self.c['relation']['mode']);self.calls+=1;self.U_packets+=len(rx);self.nonzero_sum+=facts['nonzero_relation_fraction']
  record={k:v for k,v in facts.items() if not torch.is_tensor(v)};record.update(step=step,call=self.calls,bank_teacher_step=self.bank_step,bank_age_updates=step-1-self.bank_step,role='U_s',U_truth_read=False,lambda_relation=self.c['relation']['lambda_relation'],lambda_response=self.c['relation']['lambda_response'],relation_loss=float(raw['relation'].detach()),response_loss=float(raw['response'].detach()),weighted_loss=float(total.detach()))
  if step in (7327,7548,8880,11100):
   head=classifier(model);named=[(n,p) for n,p in model.named_parameters() if n.startswith('id_backbone.') and p.requires_grad and p is not head.weight];params=[p for _,p in named]
   losses=dict(L=identity_loss,U_CE=.16*pseudo_loss,U_entropy=.01*entropy_loss,aux=total,relation=raw['relation'],response=raw['response'])
   gradients={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in losses.items()};norms={k:sum(float(g.detach().double().square().sum()) for g in grads if g is not None)**.5 for k,grads in gradients.items()}
   gL,gA=gradients['L'],gradients['aux'];dot=sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(gL,gA) if a is not None and b is not None);ratio=norms['aux']/max(norms['L'],1e-12)
   if self.c['relation']['mode']!='ZERO':self.criterion_values.append(ratio)
   w=torch.autograd.grad(total,[head.weight,head.gain,model.id_backbone.encoder.response_gain],retain_graph=True,allow_unused=True)
   if any(g is not None for g in w):raise ValueError('Auxiliary modified classW or gain derivatives')
   audit=dict(feature_gradient_norms=norms,aux_to_L_feature_gradient=ratio,aux_L_cosine=dot/(norms['L']*norms['aux']) if norms['L']*norms['aux'] else None,groups={k:gradient_summary(named,grads) for k,grads in gradients.items()},class_W_aux_gradient=None,PA_gain_aux_gradient=None,ref_gain_aux_gradient=None)
   record['actual_training_gradient']=audit;self.pending_update=dict(step=step,named=named,before=[p.detach().clone() for p in params],audit=audit)
  if self.log is None:self.log=gzip.open(self.out/'relation_steps.jsonl.gz','xt',encoding='utf-8')
  self.log.write(json.dumps(record,allow_nan=False)+'\n')
  return total,{'relation/loss':raw['relation'].detach(),'relation/response_loss':raw['response'].detach(),'relation/nonzero_fraction':facts['nonzero_relation_fraction'],'relation/teacher_entropy':facts['teacher_entropy_normalized'],'relation/teacher_response_norm':facts['teacher_response_L2'],'relation/all_U_count':float(len(rx))}
 @contextmanager
 def native_scope(self,n):
  builder=n._build_ssdg_wisig_data;context=n.compute_core_losses;step_method=torch.optim.AdamW.step;clip_method=torch.nn.utils.clip_grad_norm_
  def data(*args,**kwargs):
   ctx=builder(*args,**kwargs)
   from .sampler import replace
   ctx['unlabeled_loader']=replace(ctx['unlabeled_loader'],'PAIRS');self.prepare(ctx);return ctx
  def losses(*args,**kwargs):
   result=context(*args,**kwargs)
   if not hasattr(self.model,'_source_unlabeled_feature_hook'):self.model._source_unlabeled_feature_hook=self.auxiliary
   return result
  def clip(parameters,max_norm,*args,**kwargs):
   value=clip_method(parameters,max_norm,*args,**kwargs)
   if self.pending_update is not None:self.pending_update['clipping']=dict(pre_clipping_global_norm=float(value),max_norm=float(max_norm),predicted_global_factor=min(1.,float(max_norm)/(float(value)+1e-6)))
   return value
  def actual_step(optimizer,*args,**kwargs):
   pending=self.pending_update
   # Only the actual model optimizer owning EVERY monitored identity param.
   if pending is not None and not {id(p) for _,p in pending['named']}<={id(p) for g in optimizer.param_groups for p in g['params']}:pending=None
   before_grad=gradient_summary(pending['named'],[p.grad for _,p in pending['named']]) if pending else None
   result=step_method(optimizer,*args,**kwargs)
   if pending:
    movement=gradient_summary(pending['named'],[p.detach()-q for (_,p),q in zip(pending['named'],pending['before'])]);entry=dict(step=pending['step'],actual_AdamW_step=True,clipping=pending.get('clipping'),post_clipping_total_gradients=before_grad,parameter_update_groups=movement,head_fit_updates=0)
    with (self.out/'actual_parameter_updates.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
    self.pending_update=None
   return result
  n._build_ssdg_wisig_data=data;n.compute_core_losses=losses;torch.optim.AdamW.step=actual_step;torch.nn.utils.clip_grad_norm_=clip
  try:
   with super().native_scope(n):yield
  finally:
   n._build_ssdg_wisig_data=builder;n.compute_core_losses=context;torch.optim.AdamW.step=step_method;torch.nn.utils.clip_grad_norm_=clip_method
   if hasattr(getattr(self,'model',None),'_source_unlabeled_feature_hook'):del self.model._source_unlabeled_feature_hook
   if self.log is not None:self.log.close()
 def epoch(self,model,ctx,rows,out,device):
  super().epoch(model,ctx,rows,out,device)
  if self.log is not None:self.log.flush()
  history=self.U_sampler.history
  if any(r['generated_count']!=56700 or r['delivered_count'] not in (0,56700) for r in history):raise ValueError('PAIR unchanged full physical U budget differs')
  d.write(out/'relation_execution.json',dict(auxiliary_calls=self.calls,U_packet_observations=self.U_packets,all_U_used=True,anchor_refresh_calls=self.refreshes,extra_L_forward_packets=self.refreshes*180,head_fit_updates=0,nonzero_relation_fraction_mean=self.nonzero_sum/self.calls if self.calls else None,actual_aux_L_feature_gradient_ratios=self.criterion_values,U_sampling=history,U_truth_read=False,target_access=False,teacher_intervention=False,default_joint_classifier_unchanged=True))
