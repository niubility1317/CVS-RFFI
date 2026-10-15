"""Physical-unit Sinc policies, reusing cvs_sinc_units50/optimizer.py.
CENTER_UNITS maps one physical AdamW update; no product of historical multipliers.
"""
from contextlib import contextmanager
import math
import torch
PREFIX='id_backbone.encoder.core.id_backbone.sinc.'
NAMES=(PREFIX+'low_hz_',PREFIX+'band_hz_')
QNAMES=(PREFIX+'low_coordinate',PREFIX+'band_coordinate')
MODES=('STANDARD','NORMALIZED_UNITS','CENTER_UNITS')

def specification(model,params,mode,options):
 if mode not in MODES:raise ValueError('Unregistered optimizer unit mode')
 flat=list(params);named=dict(model.named_parameters());sinc_module=model.get_submodule(PREFIX[:-1]);centered=type(sinc_module).__name__=='CoordinateSinc'
 names=QNAMES if centered else NAMES;sinc=[named[n] for n in names]
 if not flat or any(not isinstance(p,torch.nn.Parameter) for p in flat) or len({id(p) for p in flat})!=len(flat):raise ValueError('Native flat deduplicated parameters required')
 if any(not p.requires_grad or p.shape!=(24,1) for p in sinc) or not named['id_backbone.encoder.core.id_backbone.cls_head.weight'].requires_grad:raise ValueError('Sinc/body/default-head joint training required')
 if {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in flat}:raise ValueError('All and only jointly trainable parameters required')
 if named['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape!=(6,160):raise ValueError('Default head changed')
 fs=float(sinc_module.sample_rate)
 if fs!=25000000. or not math.isfinite(fs):raise ValueError('Fixed ManySig sampling units required')
 if mode=='CENTER_UNITS' and (not centered or sinc_module.scale_hz!=25000. or sinc_module.mode!='CENTER_25K'):raise ValueError('Unified mode requires centered 25k leaves')
 if mode=='NORMALIZED_UNITS' and centered:raise ValueError('Raw-unit policy cannot be stacked on coordinate policy')
 if options.get('maximize',False) or options.get('differentiable',False) or options.get('amsgrad',False):raise ValueError('Native AdamW options differ')
 lr=float(options.get('lr',.001));eps=float(options.get('eps',1e-8));wd=float(options.get('weight_decay',.01))
 if lr<=0 or eps<=0 or wd<0 or not all(math.isfinite(x) for x in (lr,eps,wd)):raise ValueError('Invalid optimizer hyperparameters')
 scale=25000. if centered else 1.;origins=[sinc_module.low_origin_hz,sinc_module.band_origin_hz] if centered else [None,None]
 policy=dict(mode=mode,multiplier=fs/scale if mode=='CENTER_UNITS' else fs if mode=='NORMALIZED_UNITS' else 1.,base_eps=eps,base_weight_decay=wd,sinc=sinc,names=names,scale=scale,origins=origins)
 if mode=='STANDARD':return flat,policy
 ids={id(p) for p in sinc};other=[p for p in flat if id(p) not in ids]
 if not other:raise ValueError('Missing joint body/head')
 return [dict(params=other),dict(params=sinc,lr=lr*policy['multiplier'],eps=eps/policy['multiplier'],weight_decay=0. if mode=='CENTER_UNITS' else wd/fs)],policy

def attach(opt,policy):
 opt._sinc_unit_policy=policy;opt._sinc_unit_steps=0;opt._sinc_unit_lr_records=[];opt._sinc_physical_clip_calls=0;opt._sinc_pre_step={}
 def pre(opt,args,kwargs):
  base=float(opt.param_groups[0]['lr'])
  if not math.isfinite(base) or base<=0:raise ValueError('Native cosine alpha invalid')
  # Small actual pre-update witnesses include physical decay, not only adaptive motion.
  opt._sinc_pre_step={n:p.detach().clone() for n,p in zip(policy['names'],policy['sinc'])}
  if policy['mode']!='STANDARD':
   group=opt.param_groups[1]
   if len(opt.param_groups)!=2 or [id(p) for p in group['params']]!=[id(p) for p in policy['sinc']]:raise ValueError('Actual Sinc optimizer grouping changed')
   group.update(lr=base*policy['multiplier'],eps=policy['base_eps']/policy['multiplier'],weight_decay=0. if policy['mode']=='CENTER_UNITS' else policy['base_weight_decay']/25000000.)
  if policy['mode']=='CENTER_UNITS':
   if opt._sinc_physical_clip_calls!=opt._sinc_unit_steps+1:raise ValueError('Unified step missing one physical global clip')
   with torch.no_grad():
    for q,origin in zip(policy['sinc'],policy['origins']):
     if q.grad is not None:q.add_(q+origin/policy['scale'],alpha=-base*policy['base_weight_decay'])
  group=opt.param_groups[-1]
  opt._sinc_unit_lr_records.append(dict(step=opt._sinc_unit_steps+1,base_lr=base,sinc_lr=float(group['lr']),sinc_eps=float(group['eps']),sinc_weight_decay=float(group['weight_decay']),physical_decay_alpha_lambda=base*policy['base_weight_decay'] if policy['mode']=='CENTER_UNITS' else None,physical_clip_calls=opt._sinc_physical_clip_calls))
 def post(opt,args,kwargs):opt._sinc_unit_steps+=1
 opt.register_step_pre_hook(pre);opt.register_step_post_hook(post);return opt

def build(model,params,mode,*,optimizer_class=torch.optim.AdamW,**options):
 groups,policy=specification(model,params,mode,options);return attach(optimizer_class(groups,**options),policy)

def clip_physical(opt,parameters,max_norm,*args,clip_impl=None,**kwargs):
 """Use physical Hz gradients with all other parameters in one original clip5."""
 impl=clip_impl or torch.nn.utils.clip_grad_norm_;params=list(parameters);policy=opt._sinc_unit_policy
 if policy['mode']!='CENTER_UNITS':return impl(params,max_norm,*args,**kwargs)
 if float(max_norm)!=5. or {id(p) for p in params}!={id(p) for g in opt.param_groups for p in g['params']}:raise ValueError('Unified global clip must cover original model at clip5')
 with torch.no_grad():
  for q in policy['sinc']:
   if q.grad is not None:q.grad.div_(policy['scale'])
  try:total=impl(params,max_norm,*args,**kwargs)
  finally:
   for q in policy['sinc']:
    if q.grad is not None:q.grad.mul_(policy['scale'])
 opt._sinc_physical_clip_calls+=1;return total

@contextmanager
def native_scope(owner,mode):
 if mode not in MODES:raise ValueError('Unregistered optimizer mode')
 original=torch.optim.AdamW;original_clip=torch.nn.utils.clip_grad_norm_
 class ScopedAdamW(original):
  def __init__(self,params,**options):
   groups,policy=specification(owner.model,params,mode,options);super().__init__(groups,**options);attach(self,policy)
   if getattr(owner,'optimizer',None) is not None:raise ValueError('More than one native optimizer')
   owner.optimizer=self
   if hasattr(owner,'optimizer_created'):owner.optimizer_created(self)
 def wrapped(parameters,max_norm,*args,**kwargs):
  if mode!='CENTER_UNITS':return original_clip(parameters,max_norm,*args,**kwargs)
  if owner.optimizer is None:raise ValueError('Physical clipping before native optimizer')
  return clip_physical(owner.optimizer,parameters,max_norm,*args,clip_impl=original_clip,**kwargs)
 torch.optim.AdamW=ScopedAdamW;torch.nn.utils.clip_grad_norm_=wrapped
 try:yield
 finally:torch.optim.AdamW=original;torch.nn.utils.clip_grad_norm_=original_clip
