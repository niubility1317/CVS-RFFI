"""Optimizer-only Sinc unit policy; original network and parameter tensors retained."""
from contextlib import contextmanager
import math
import torch
PREFIX='id_backbone.encoder.core.id_backbone.sinc.'
NAMES=(PREFIX+'low_hz_',PREFIX+'band_hz_')
MODES=('STANDARD','NORMALIZED_UNITS')

def specification(model,params,mode,options):
 if mode not in MODES:raise ValueError('Unregistered optimizer unit mode')
 flat=list(params)
 if not flat or any(not isinstance(p,torch.nn.Parameter) for p in flat) or len({id(p) for p in flat})!=len(flat):raise ValueError('Native flat deduplicated parameter list required')
 named=dict(model.named_parameters());sinc=[named[n] for n in NAMES]
 if any(not p.requires_grad or p.shape!=(24,1) for p in sinc) or not named['id_backbone.encoder.core.id_backbone.cls_head.weight'].requires_grad:raise ValueError('Original Sinc/body/default-head joint training required')
 if {id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in flat}:raise ValueError('Native optimizer must contain all and only jointly trainable parameters')
 if named['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape!=(6,160):raise ValueError('Default head changed')
 fs=float(model.id_backbone.encoder.core.id_backbone.sinc.sample_rate)
 if fs!=25000000. or not math.isfinite(fs):raise ValueError('Existing fixed ManySig sampling units required')
 if options.get('maximize',False) or options.get('differentiable',False) or options.get('amsgrad',False):raise ValueError('Native AdamW options differ')
 lr=float(options.get('lr',.001));eps=float(options.get('eps',1e-8));wd=float(options.get('weight_decay',.01))
 if lr<=0 or eps<=0 or wd<0 or not all(math.isfinite(x) for x in (lr,eps,wd)):raise ValueError('Invalid native optimizer hyperparameters')
 if mode=='STANDARD':return flat,dict(mode=mode,multiplier=1.,base_eps=eps,base_weight_decay=wd,sinc=sinc)
 ids={id(p) for p in sinc};other=[p for p in flat if id(p) not in ids]
 if not other:raise ValueError('Missing jointly trained body/head')
 groups=[dict(params=other),dict(params=sinc,lr=lr*fs,eps=eps/fs,weight_decay=wd/fs)]
 return groups,dict(mode=mode,multiplier=fs,base_eps=eps,base_weight_decay=wd,sinc=sinc)

def attach(optimizer,policy):
 optimizer._sinc_unit_policy=policy;optimizer._sinc_unit_steps=0;optimizer._sinc_unit_lr_records=[]
 def pre(opt,args,kwargs):
  base=float(opt.param_groups[0]['lr'])
  if not math.isfinite(base) or base<=0:raise ValueError('Native cosine LR invalid')
  if policy['mode']=='NORMALIZED_UNITS':
   group=opt.param_groups[1]
   if len(opt.param_groups)!=2 or [id(p) for p in group['params']]!=[id(p) for p in policy['sinc']]:raise ValueError('Actual Sinc optimizer grouping changed')
   group.update(lr=base*policy['multiplier'],eps=policy['base_eps']/policy['multiplier'],weight_decay=policy['base_weight_decay']/policy['multiplier'])
  group=opt.param_groups[-1]
  opt._sinc_unit_lr_records.append(dict(step=opt._sinc_unit_steps+1,base_lr=base,sinc_lr=float(group['lr']),sinc_eps=float(group['eps']),sinc_weight_decay=float(group['weight_decay'])))
 def post(opt,args,kwargs):opt._sinc_unit_steps+=1
 optimizer.register_step_pre_hook(pre);optimizer.register_step_post_hook(post)
 return optimizer

def build(model,params,mode,*,optimizer_class=torch.optim.AdamW,**options):
 groups,policy=specification(model,params,mode,options)
 return attach(optimizer_class(groups,**options),policy)

@contextmanager
def native_scope(owner,mode):
 """The caller supplies owner.model before the one native optimizer is constructed."""
 if mode not in MODES:raise ValueError('Unregistered optimizer unit mode')
 original=torch.optim.AdamW
 class ScopedAdamW(original):
  def __init__(self,params,**options):
   groups,policy=specification(owner.model,params,mode,options)
   super().__init__(groups,**options);attach(self,policy)
   if getattr(owner,'optimizer',None) is not None:raise ValueError('More than one native optimizer')
   owner.optimizer=self
   if hasattr(owner,'optimizer_created'):owner.optimizer_created(self)
 torch.optim.AdamW=ScopedAdamW
 try:yield
 finally:torch.optim.AdamW=original
