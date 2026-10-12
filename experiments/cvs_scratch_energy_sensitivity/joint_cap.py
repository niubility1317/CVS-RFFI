"""One detached scalar caps the already-combined Fishr gradient by group.

This is a bound on loss gradients at the current parameters. It is not a bound
on the AdamW update: optimizer moments, learning rates and weight decay differ.
"""
import math
import time
import torch

GROUPS=('shallow','middle','rest')


def parameter_groups(model):
    """Disjoint complete partition of every trainable base parameter, including Hz."""
    named=tuple((name,p) for name,p in model.base.named_parameters() if p.requires_grad)
    if not named:
        raise ValueError('Joint cap requires trainable model.base parameters')
    grouped={key:[] for key in GROUPS}
    for name,p in named:
        padded='.'+name+'.'
        group='shallow' if '.time_fuse.' in padded else 'middle' if '.t2.' in padded else 'rest'
        grouped[group].append((name,p))
    members=[id(p) for rows in grouped.values() for _,p in rows]
    if len(members)!=len(set(members)) or set(members)!={id(p) for _,p in named}:
        raise AssertionError('Joint cap parameter partition is not complete and disjoint')
    return named,grouped


def inactive_detail(reason):
    return dict(enabled=False,applied=False,reason=reason,scalar=1.,groups={},
        gradient_evaluations=0,seconds=0.,scalar_detached=True,
        scope='current loss gradient; not an AdamW parameter-displacement bound')


def _norms(loss,named,grouped):
    if loss.ndim!=0 or loss.is_complex() or not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError('Joint cap needs a finite real scalar loss')
    parameters=tuple(p for _,p in named)
    gradients=(torch.autograd.grad(loss,parameters,retain_graph=True,create_graph=False,allow_unused=True)
               if loss.requires_grad else (None,)*len(parameters))
    by_id={id(p):g for p,g in zip(parameters,gradients)}
    norms={}
    for key,rows in grouped.items():
        # abs() is essential for complex coordinates; g.square() is not a norm.
        squared=[by_id[id(p)].detach().abs().to(torch.float64).square().sum()
                 for _,p in rows if by_id[id(p)] is not None]
        value=math.sqrt(float(torch.stack(squared).sum())) if squared else 0.
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite joint cap '+key+' gradient norm')
        norms[key]=value
    return norms,int(loss.requires_grad)


def apply_joint_cap(model,identity_loss,raw_fishr,config):
    """Return scaled R and JSON-safe measurements; no parameter.grad or RNG writes.

    identity_loss is the weighted sum of all non-Fishr training terms, excluding
    audit-only KL decomposition keys. raw_fishr already combines selected layers,
    fixed calibration coefficients, ramp, and the fixed denominator of two.
    """
    if not config['enabled']:
        return raw_fishr,inactive_detail('disabled_by_policy')
    started=time.perf_counter();rho=float(config['rho']);eps=float(config['eps'])
    if not math.isfinite(rho+eps) or rho<0 or eps<=0:
        raise ValueError('Invalid joint gradient cap rho/epsilon')
    named,grouped=parameter_groups(model)
    identity,n_id=_norms(identity_loss,named,grouped)
    regularizer,n_r=_norms(raw_fishr,named,grouped)
    bounds={key:rho*identity[key]/(regularizer[key]+eps) if regularizer[key]>0 else None for key in GROUPS}
    scalar=min([1.]+[value for value in bounds.values() if value is not None])
    if not math.isfinite(scalar) or not 0<=scalar<=1:
        raise FloatingPointError('Invalid joint gradient cap scalar')
    # Python float intentionally disconnects norm computations from this backward.
    effective=raw_fishr*scalar
    if not bool(torch.isfinite(effective.detach())):
        raise FloatingPointError('Nonfinite capped Fishr loss')
    detail=dict(enabled=True,applied=True,reason=None,scalar=scalar,scalar_detached=True,rho=rho,eps=eps,
        groups={},gradient_evaluations=n_id+n_r,seconds=time.perf_counter()-started,
        identity_weighted_loss=float(identity_loss.detach()),raw_weighted_fishr=float(raw_fishr.detach()),
        effective_weighted_fishr=float(effective.detach()),all_regularizer_groups_zero=all(v==0 for v in regularizer.values()),
        zero_identity_positive_regularizer_groups=[key for key in GROUPS if identity[key]==0 and regularizer[key]>0],
        active_bound_groups=[key for key in GROUPS if bounds[key] is not None and bounds[key]==scalar and scalar<1],
        complete_disjoint_parameter_coverage=True,parameter_tensors=len(named),
        parameter_scalars=sum(p.numel() for _,p in named),
        gradient_definition='actual weighted non-Fishr objective versus joint weighted Fishr; no layerwise clipping',
        norm_measurement='id and raw R from autograd; capped R = detached scalar times measured raw R',
        scope='current loss gradient; not an AdamW parameter-displacement bound')
    for key,rows in grouped.items():
        detail['groups'][key]=dict(norm_id=identity[key],raw_R=regularizer[key],capped_R=scalar*regularizer[key],
            norm_bound=rho*identity[key],scalar_bound=bounds[key],regularizer_zero=regularizer[key]==0,
            identity_zero=identity[key]==0,parameter_tensors=len(rows),parameter_scalars=sum(p.numel() for _,p in rows),
            real_coordinates=sum(p.numel()*(2 if p.is_complex() else 1) for _,p in rows),
            parameter_names_sample=[name for name,_ in rows[:3]],parameter_names_truncated=len(rows)>3)
    return effective,detail
