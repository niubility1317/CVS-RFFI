"""Differentiable source membership objectives and labeled-anchor protection."""
import math
import torch
from torch.nn import functional as F


def _finite(value,name):
    if not isinstance(value,torch.Tensor) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(name+' must be a finite floating-point tensor')
    return value


def prototypes(z_support,slot_order,held_class,arm):
    _finite(z_support,'z_support')
    if z_support.shape!=(6,16,64):raise ValueError('Source support must have shape [6,16,64]')
    if arm not in ('S0','S1','M') or type(held_class) is not int or held_class not in range(6):
        raise ValueError('Fixed source training arm and held class required')
    order=torch.as_tensor(slot_order,device=z_support.device)
    if order.dtype!=torch.int64 or order.shape!=(6,) or not torch.equal(order.sort().values,torch.arange(6,device=order.device)):
        raise ValueError('slot_order must be a permutation of six source classes')
    labels=order[order!=held_class] if arm=='M' else order
    means=z_support.mean(dim=1)[labels]
    if bool((torch.linalg.vector_norm(means,dim=1)<=1e-12).any()):raise ValueError('Nonzero source prototype means required')
    return F.normalize(means,dim=1,eps=1e-12),labels


def member_loss(pair_logits,query_y,prototype_labels):
    _finite(pair_logits,'pair_logits')
    if pair_logits.ndim!=2 or pair_logits.shape[0]==0 or not 1<=pair_logits.shape[1]<=6:
        raise ValueError('Nonempty pair logits [N,K] with source K in 1..6 required')
    n,k=pair_logits.shape
    if (not isinstance(query_y,torch.Tensor) or query_y.dtype!=torch.int64 or query_y.shape!=(n,)
        or query_y.device!=pair_logits.device or bool(((query_y < -1)|(query_y>5)).any())
        or not isinstance(prototype_labels,torch.Tensor) or prototype_labels.dtype!=torch.int64
        or prototype_labels.shape!=(k,) or prototype_labels.device!=pair_logits.device
        or prototype_labels.unique().numel()!=k or bool(((prototype_labels<0)|(prototype_labels>5)).any())):
        raise ValueError('Source-only int64 labels (OE=-1) and unique source prototype labels required')
    pair_target=query_y[:,None]==prototype_labels[None,:]
    set_target=pair_target.any(dim=1)
    pair_positive=int(pair_target.sum());set_positive=int(set_target.sum())
    counts=dict(pair_positive=pair_positive,pair_negative=n*k-pair_positive,set_positive=set_positive,
                set_negative=n-set_positive,query_count=n,prototype_count=k)
    if min(counts[x] for x in ('pair_positive','pair_negative','set_positive','set_negative'))==0:
        raise ValueError('Balanced membership loss requires positive and negative pair and set examples')
    pair=F.binary_cross_entropy_with_logits(pair_logits,pair_target.to(pair_logits.dtype),reduction='none')
    set_bce=F.binary_cross_entropy_with_logits(pair_logits.max(dim=1).values,set_target.to(pair_logits.dtype),reduction='none')
    pair_loss=.5*(pair[pair_target].mean()+pair[~pair_target].mean())
    set_loss=.5*(set_bce[set_target].mean()+set_bce[~set_target].mean())
    return dict(pair_bce=pair_loss,set_bce=set_loss,member_loss=.5*pair_loss+.5*set_loss,counts=counts)


def kd_loss(student_logits,teacher_logits,temperature=2.):
    _finite(student_logits,'student_logits');_finite(teacher_logits,'teacher_logits')
    if (student_logits.ndim!=2 or student_logits.shape[0]==0 or student_logits.shape[1]<2
        or student_logits.shape!=teacher_logits.shape or student_logits.device!=teacher_logits.device
        or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature<=0):
        raise ValueError('Aligned logits and a positive finite KD temperature required')
    return F.kl_div(F.log_softmax(student_logits/temperature,dim=1),F.softmax(teacher_logits.detach()/temperature,dim=1),reduction='batchmean')*(temperature**2)


def protected_gradients(anchor_loss,member_loss_weighted,namedparams,auxiliary_prefix='open_world.relation.'):
    """Return first-order grads in input order, retaining None for unused params.

    Projection/cap coefficients are computed jointly across all shared parameters
    in FP64. Relation parameters receive unprojected summed gradients. The
    caller supplies auxiliary_prefix to match its actual relation-module name.
    """
    for loss,name in ((anchor_loss,'anchor_loss'),(member_loss_weighted,'member_loss_weighted')):
        _finite(loss,name)
        if loss.ndim!=0:raise ValueError(name+' must be scalar')
    named=list(namedparams)
    if len({n for n,p in named})!=len(named) or len({id(p) for n,p in named})!=len(named):
        raise ValueError('Named parameters must be unique')
    active=[(i,n,p) for i,(n,p) in enumerate(named) if p.requires_grad]
    parameters=tuple(p for _,_,p in active)
    def gradients(loss,retain):
        if not parameters or not loss.requires_grad:return (None,)*len(parameters)
        return torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=retain)
    gl=gradients(anchor_loss,True);go=gradients(member_loss_weighted,False)
    def finite(grads):
        checks=[torch.isfinite(g).all() for g in grads if g is not None]
        return not checks or bool(torch.stack(checks).all())
    for label,grads in (('anchor',gl),('member',go)):
        if not finite(grads):raise ValueError('Nonfinite '+label+' gradient')
    shared=[j for j,(_,name,p) in enumerate(active) if not name.startswith(auxiliary_prefix)]
    def norm2(values):
        terms=[g.detach().double().square().sum() for g in values if g is not None]
        return float(torch.stack(terms).sum()) if terms else 0.
    def dot(left,right):
        terms=[x.detach().double().mul(y.detach().double()).sum() for x,y in zip(left,right) if x is not None and y is not None]
        return float(torch.stack(terms).sum()) if terms else 0.
    left=[gl[j] for j in shared];right=[go[j] for j in shared]
    l2=norm2(left);o2=norm2(right);before=dot(left,right);conflict=before<0 and l2>0
    coefficient=before/l2 if conflict else 0.
    protected=[]
    for j in shared:
        if gl[j] is None and go[j] is None:protected.append(None);continue
        extra=go[j] if go[j] is not None else torch.zeros_like(active[j][2])
        if l2==0:extra=torch.zeros_like(extra)
        elif conflict and gl[j] is not None:extra=extra-coefficient*gl[j]
        protected.append(extra)
    projected2=norm2(protected)
    scale=min(1.,math.sqrt(l2/projected2)) if projected2>0 else 1.
    protected=[g*scale if g is not None else None for g in protected]
    shared_grad=dict(zip(shared,protected));result=[None]*len(named)
    for j,(index,name,p) in enumerate(active):
        extra=shared_grad[j] if j in shared_grad else go[j]
        result[index]=(gl[j]+extra if gl[j] is not None and extra is not None else gl[j] if extra is None else extra)
    if not finite(result):raise ValueError('Nonfinite merged gradient')
    diagnostics=dict(shared_anchor_norm=math.sqrt(l2),shared_member_norm_before=math.sqrt(o2),
        shared_member_norm_projected=math.sqrt(projected2),shared_member_norm_after=math.sqrt(norm2(protected)),
        shared_dot_before=before,shared_dot_after=dot(left,protected),conflict_projected=conflict,
        cap_scale=scale,anchor_zero=l2==0,shared_parameter_count=len(shared),auxiliary_parameter_count=len(active)-len(shared),
        unused_parameters=[name for j,(_,name,p) in enumerate(active) if gl[j] is None and go[j] is None],
        anchor_unused=[name for j,(_,name,p) in enumerate(active) if gl[j] is None],
        member_unused=[name for j,(_,name,p) in enumerate(active) if go[j] is None],
        frozen_parameters=[name for name,p in named if not p.requires_grad],
        anchor_constant=not anchor_loss.requires_grad,member_constant=not member_loss_weighted.requires_grad)
    return tuple(result),diagnostics
