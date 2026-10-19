"""Reset AdamW for frozen-head F or adapter-only continuation."""
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
    sinc = model.get_submodule(PREFIX[:-1])
    qs = [named[k] for k in QNAMES]
    if c['adapter']['enabled']:
        if any(p.requires_grad for p in qs) or sum(p.numel() for p in flat) != c['adapter']['parameters']:
            raise ValueError('Adapter row must freeze Sinc and train exactly the new adapter parameters')
        # The model applies the named trainable scope; require every original
        # parent parameter to remain frozen, rather than guessing adapter names.
        parent_keys = set(getattr(model, '_dta_original_parameter_names', ()))
        if not parent_keys or any(named[k].requires_grad for k in parent_keys):
            raise ValueError('Adapter-only optimizer requires the recorded frozen original F scope')
        return flat, dict(mode='ADAPTER_ONLY',sinc=[],names=[],base_eps=eps,base_weight_decay=wd,
                          multiplier=0.,scale=25000.,origins=[])
    if (type(sinc).__name__ != 'CoordinateSinc' or sinc.mode != 'CENTER_25K'
            or sinc.sample_rate != 25000000. or sinc.scale_hz != 25000.
            or any(not p.requires_grad or p.shape != (24,1) for p in qs)):
        raise ValueError('Original F continuation requires trainable physical CENTER_25K Sinc')
    ids = {id(p) for p in qs}
    other = [p for p in flat if id(p) not in ids]
    if not other:
        raise ValueError('Original F body must remain trainable')
    policy = dict(mode='CENTER_UNITS',sinc=qs,names=QNAMES,base_eps=eps,base_weight_decay=wd,
        multiplier=1000.,scale=25000.,origins=[sinc.low_origin_hz,sinc.band_origin_hz])
    return [dict(params=other),dict(params=qs,lr=lr*1000.,eps=eps/1000.,weight_decay=0.)],policy


def attach(opt, policy):
    if policy['mode'] == 'CENTER_UNITS':
        from experiments.cvs_rdrop_sinc_units.optimizer import attach as physical_attach
        return physical_attach(opt,policy)
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
        if opt._sinc_unit_policy['mode']=='CENTER_UNITS':
            from experiments.cvs_rdrop_sinc_units.optimizer import clip_physical
            return clip_physical(opt,params,max_norm,*args,clip_impl=clip,**kwargs)
        result=clip(params,max_norm,*args,**kwargs)
        opt._sinc_physical_clip_calls+=1
        return result
    torch.optim.AdamW=ScopedAdamW
    torch.nn.utils.clip_grad_norm_=clipped
    try:yield
    finally:
        torch.optim.AdamW=original
        torch.nn.utils.clip_grad_norm_=clip
