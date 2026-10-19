"""Reset ordinary AdamW for the raw-IQ adapter only; original F/head frozen."""
from contextlib import contextmanager
import math
import torch

PREFIX = 'id_backbone.encoder.core.id_backbone.sinc.'
QNAMES = (PREFIX+'low_coordinate', PREFIX+'band_coordinate')
HEAD = 'id_backbone.encoder.core.id_backbone.cls_head.'


def specification(model, params, c, options):
    flat = list(params)
    named = dict(model.named_parameters())
    if (not flat or any(not isinstance(p, torch.nn.Parameter) for p in flat)
            or len({id(p) for p in flat}) != len(flat)
            or {id(p) for p in flat} != {id(p) for p in model.parameters() if p.requires_grad}
            or any(p.requires_grad for k,p in named.items() if k.startswith(HEAD))
            or named[HEAD+'weight'].shape != (6,160)):
        raise ValueError('All and only trainable parameters with the original frozen head required')
    lr, eps, wd = (float(options.get(k, default)) for k,default in
                   (('lr',.001),('eps',1e-8),('weight_decay',.01)))
    if (lr != c['optimizer']['lr'] or wd != c['optimizer']['weight_decay'] or eps <= 0
            or not all(math.isfinite(v) for v in (lr,eps,wd))
            or any(options.get(k,False) for k in ('maximize','differentiable','amsgrad'))):
        raise ValueError('Fixed reset AdamW settings differ')
    qs = [named[k] for k in QNAMES]
    if (not c['adapter']['enabled'] or c['original_F_trainable'] or any(p.requires_grad for p in qs)
            or sum(p.numel() for p in flat)!=4256 or c['adapter']['parameters']!=4256):
        raise ValueError('Only the 4256 raw-IQ adapter parameters may train; original Sinc is frozen')
    parent_keys=set(getattr(model,'_dta_original_parameter_names',()))
    adapter_names=set(getattr(model,'_dta_adapter_names',()))
    if (not parent_keys or any(named[k].requires_grad for k in parent_keys)
            or not adapter_names or {id(named[k]) for k in adapter_names}!={id(p) for p in flat}):
        raise ValueError('Optimizer must contain only the recorded new raw-IQ adapter scope')
    return flat,dict(mode='ADAPTER_ONLY',architecture='RAW_IQ_TWO_PRE_GN',sinc=[],names=[],
        base_eps=eps,base_weight_decay=wd,multiplier=0.,scale=25000.,origins=[])


def attach(opt, policy):
    opt._sinc_unit_policy=policy
    opt._sinc_unit_steps=0
    opt._sinc_physical_clip_calls=0
    opt._sinc_unit_lr_records=[]
    def pre(opt,args,kwargs):
        if opt._sinc_physical_clip_calls != opt._sinc_unit_steps+1:
            raise ValueError('One global adapter clip is required per native optimizer update')
        opt._sinc_unit_lr_records.append(dict(step=opt._sinc_unit_steps+1,
            base_lr=float(opt.param_groups[0]['lr']),sinc_lr=0.,sinc_eps=None,
            sinc_weight_decay=0.,physical_decay_alpha_lambda=0.,
            physical_clip_calls=opt._sinc_physical_clip_calls,mode='ADAPTER_ONLY',Sinc_updates=0))
    def post(opt,args,kwargs):opt._sinc_unit_steps+=1
    opt.register_step_pre_hook(pre);opt.register_step_post_hook(post)
    return opt


@contextmanager
def native_scope(owner):
    original=torch.optim.AdamW
    clip=torch.nn.utils.clip_grad_norm_
    class ScopedAdamW(original):
        def __init__(self,params,**options):
            groups,policy=specification(owner.model,params,owner.c,options)
            super().__init__(groups,**options)
            attach(self,policy)
            if owner.optimizer is not None:
                raise ValueError('Exactly one native reset optimizer is supported')
            owner.optimizer=self
            owner.optimizer_created(self)
    def clipped(parameters,max_norm,*args,**kwargs):
        opt=owner.optimizer
        if opt is None:raise ValueError('Clipping before optimizer construction')
        params=list(parameters)
        if float(max_norm)!=owner.c['optimizer']['clip'] or {id(p) for p in params}!={id(p) for g in opt.param_groups for p in g['params']}:
            raise ValueError('Native clip5 must cover exactly the active optimizer parameters')
        result=clip(params,max_norm,*args,**kwargs)
        opt._sinc_physical_clip_calls+=1
        return result
    torch.optim.AdamW=ScopedAdamW
    torch.nn.utils.clip_grad_norm_=clipped
    try:yield
    finally:
        torch.optim.AdamW=original
        torch.nn.utils.clip_grad_norm_=clip
