import torch
from . import design as d
from experiments.pamf_r46_gradient_geometry.gradients import parameter_groups as diagnostic_groups

SCALARS=('base_norm','moment_norm','raw_norm','applied_norm','raw_base_dot','raw_moment_dot',
    'applied_base_dot','applied_moment_dot','conflict','projected','base_cos_valid','base_cos','moment_cos_valid','moment_cos')


def parameter_groups(model):
    params,groups=diagnostic_groups(model)
    return params,{name:groups[name] for name in d.GRADIENT_GROUPS}


def combine_gradients(base,moment,parameters,groups,policy):
    if policy not in ('sum','symmetric_module_projection'):raise ValueError('Registered gradient policy required')
    if len(base)!=len(parameters) or len(moment)!=len(parameters) or not parameters or not groups:
        raise ValueError('Complete two-objective parameter gradients required')
    indices=[i for group in groups.values() for i in group]
    if any(not group for group in groups.values()) or sorted(indices)!=list(range(len(parameters))):
        raise ValueError('Disjoint complete unique parameter groups required')
    for values in (base,moment):
        if any(g is not None and (g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device) for p,g in zip(parameters,values)):
            raise ValueError('Gradient shape,dtype and device must match parameters')
    combined=[None]*len(parameters);packed=[];finite=[];names=list(groups)
    for name in names:
        take=groups[name]
        def flatten(values):
            return torch.cat([torch.zeros_like(parameters[i]).flatten() if values[i] is None else values[i].detach().flatten() for i in take]).double()
        a,b=flatten(base),flatten(moment);a2=a.square().sum();b2=b.square().sum();c=torch.dot(a,b)
        valid=(a2>d.NORM_FLOOR**2)&(b2>d.NORM_FLOOR**2);conflict=valid&(c<0)
        ka=torch.where(conflict,c/a2.clamp_min(d.NORM_FLOOR**2),torch.zeros_like(c))
        kb=torch.where(conflict,c/b2.clamp_min(d.NORM_FLOOR**2),torch.zeros_like(c))
        raw=a+b;proposed=(1-ka)*a+(1-kb)*b
        selected=proposed if policy=='symmetric_module_projection' else raw
        offset=0;pieces=[]
        for i in take:
            size=parameters[i].numel();part=selected[offset:offset+size].to(parameters[i].dtype).reshape_as(parameters[i]);offset+=size
            combined[i]=None if base[i] is None and moment[i] is None else part.detach()
            pieces.append(part.flatten().double())
        applied=torch.cat(pieces);an=a2.sqrt();bn=b2.sqrt();rn=torch.linalg.vector_norm(raw);zn=torch.linalg.vector_norm(applied)
        ad=torch.dot(a,applied);bd=torch.dot(b,applied);av=(an>d.NORM_FLOOR)&(zn>d.NORM_FLOOR);bv=(bn>d.NORM_FLOOR)&(zn>d.NORM_FLOOR)
        ac=torch.where(av,ad/(an*zn).clamp_min(d.NORM_FLOOR**2),torch.zeros_like(ad)).clamp(-1,1)
        bc=torch.where(bv,bd/(bn*zn).clamp_min(d.NORM_FLOOR**2),torch.zeros_like(bd)).clamp(-1,1)
        packed.append(torch.stack([an,bn,rn,zn,torch.dot(a,raw),torch.dot(b,raw),ad,bd,conflict.double(),
            conflict.double()*int(policy=='symmetric_module_projection'),av.double(),ac,bv.double(),bc]))
        finite.append(torch.isfinite(a).all()&torch.isfinite(b).all()&torch.isfinite(applied).all())
    values=torch.stack(packed)
    if not bool(torch.stack(finite).all()&torch.isfinite(values).all()):raise FloatingPointError('Nonfinite source gradient combination')
    stats={name:dict(zip(SCALARS,row)) for name,row in zip(names,values.detach().cpu().tolist())}
    for row in stats.values():
        for field in ('conflict','projected','base_cos_valid','moment_cos_valid'):row[field]=int(row[field])
        if not row['base_cos_valid']:row['base_cos']=None
        if not row['moment_cos_valid']:row['moment_cos']=None
    return combined,stats


def assign_gradients(model,terms,policy,grouping=None):
    parameters,groups=parameter_groups(model) if grouping is None else grouping
    if any(not p.requires_grad or p.grad is not None for p in parameters):raise ValueError('Trainable model with cleared gradients required')
    a=torch.autograd.grad(terms['base_loss'].mean(),parameters,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(terms['weighted_moment_objective'].mean(),parameters,allow_unused=True)
    combined,stats=combine_gradients(a,b,parameters,groups,policy)
    for p,g in zip(parameters,combined):p.grad=g
    return stats


def empty_epoch_stats():
    return {g:dict(updates=0,conflict_updates=0,projected_updates=0,sums={k:0. for k in SCALARS[:8]},
        base_cos_sum=0.,base_cos_valid=0,moment_cos_sum=0.,moment_cos_valid=0) for g in d.GRADIENT_GROUPS}


def add_epoch_stats(totals,current):
    for group,row in current.items():
        total=totals[group];total['updates']+=1;total['conflict_updates']+=row['conflict'];total['projected_updates']+=row['projected']
        for key in SCALARS[:8]:total['sums'][key]+=row[key]
        for key in ('base','moment'):
            total[key+'_cos_valid']+=row[key+'_cos_valid']
            if row[key+'_cos'] is not None:total[key+'_cos_sum']+=row[key+'_cos']


def finish_epoch_stats(totals):
    result={}
    for group,row in totals.items():
        n=row['updates']
        if n!=985:raise ValueError('Every source update must have gradient-policy telemetry')
        result[group]=dict(updates=n,conflict_updates=row['conflict_updates'],projected_updates=row['projected_updates'],means={k:v/n for k,v in row['sums'].items()})
        for key in ('base','moment'):
            valid=row[key+'_cos_valid'];result[group][key+'_cos_valid']=valid
            result[group][key+'_cos_mean']=row[key+'_cos_sum']/valid if valid else None
    return result
