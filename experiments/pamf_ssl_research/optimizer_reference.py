"""One disposable L-only step using the live native AdamW state and coordinates."""
from copy import deepcopy
import torch
from torch import nn
from experiments.cric_dura_comparison.f_optimizer import PREFIX,build,clip_physical


class CoordinateSinc(nn.Module):
    """Parameter and physical-unit metadata container; has no forward method."""


def container(model,named):
    observed=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if len(named)!=len(observed) or any(n!=a or p is not q for (n,p),(a,q) in zip(named,observed)):
        raise ValueError('Exact ordered jointly trainable source parameters required')
    if not named or any(p.dtype!=torch.float32 or p.device!=named[0][1].device for _,p in named):
        raise ValueError('One device and actual FP32 source parameters required')
    source=model.get_submodule(PREFIX[:-1])
    if (type(source).__name__!='CoordinateSinc' or source.mode!='CENTER_25K'
            or source.sample_rate!=25000000. or source.scale_hz!=25000.):
        raise ValueError('Actual centered Sinc units required')
    result=nn.Module()
    for name,parameter in named:
        node=result;parts=name.split('.')
        for i,part in enumerate(parts[:-1]):
            if part not in node._modules:
                node.add_module(part,CoordinateSinc() if '.'.join(parts[:i+1])==PREFIX[:-1] else nn.Module())
            node=node._modules[part]
        node.register_parameter(parts[-1],nn.Parameter(parameter.detach().clone()))
    sinc=result.get_submodule(PREFIX[:-1])
    sinc.mode=source.mode;sinc.sample_rate=source.sample_rate;sinc.scale_hz=source.scale_hz
    for name in ('low_origin_hz','band_origin_hz'):sinc.register_buffer(name,getattr(source,name).detach().clone())
    return result


@torch.no_grad()
def labeled_step(model,named,optimizer,gradients,*,clip_impl):
    """Return actual FP32 reference parameters without changing the live trajectory."""
    named=list(named)
    if len(gradients)!=len(named):raise ValueError('Complete aligned L gradients required')
    policy=getattr(optimizer,'_sinc_unit_policy',{})
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if (policy.get('mode')!='CENTER_UNITS' or len(actual)!=len(named)
            or {id(p) for p in actual}!={id(p) for _,p in named}):
        raise ValueError('Matching actual native CENTER_UNITS optimizer required')
    shadow=container(model,named);lookup=dict(shadow.named_parameters())
    params=[lookup[name] for name,_ in named]
    for (_,source),parameter,gradient in zip(named,params,gradients):
        if gradient is not None:
            if (gradient.shape!=source.shape or gradient.dtype!=source.dtype or gradient.device!=source.device
                    or not bool(torch.isfinite(gradient).all())):
                raise ValueError('Finite matching FP32 L gradients required')
            parameter.grad=gradient.detach().clone()
    # defaults may include internal Adam options absent from AdamW.__init__.
    keys=('betas','eps','weight_decay','lr','amsgrad','maximize','foreach','capturable','differentiable','fused')
    options={key:optimizer.defaults[key] for key in keys if key in optimizer.defaults}
    options.update(lr=float(optimizer.param_groups[0]['lr']),eps=policy['base_eps'],weight_decay=policy['base_weight_decay'])
    reference=build(shadow,params,'CENTER_UNITS',**options)
    # deepcopy avoids moment aliases; load maps each group to independent parameters.
    reference.load_state_dict(deepcopy(optimizer.state_dict()))
    norm=clip_physical(reference,params,5.,clip_impl=clip_impl)
    if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite L-only physical norm')
    reference.step()
    if reference._sinc_unit_steps!=1 or reference._sinc_physical_clip_calls!=1:
        raise ValueError('Exactly one disposable physical clip and reference update required')
    values=tuple(p.detach().clone() for p in params)
    if any(not bool(torch.isfinite(p).all()) for p in values):raise ValueError('Nonfinite L-only reference parameters')
    return values,dict(reference_optimizer_steps=1,reference_physical_clip_calls=1,copied_existing_moments=True,
                       live_optimizer_updates=0,model_forward_calls=0,physical_grad_norm_before_clip=float(norm))
