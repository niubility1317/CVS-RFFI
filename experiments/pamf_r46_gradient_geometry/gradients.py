import numpy as np
import torch
from . import design as d
from experiments.pamf_r46_moment_fidelity.models import objective


def parameter_groups(model):
    named=list(model.named_parameters());parameters=[p for _,p in named]
    groups={name:[] for name in d.GROUPS};groups['whole']=list(range(len(parameters)))
    for index,(name,_) in enumerate(named):
        key=('coarse' if name.startswith('frozen.coarse.') else 'action' if name.startswith('frozen.estimator.') else 'prior' if name.startswith('prior.') else None)
        if key is None:raise ValueError('Every unique parameter must have a registered physical module group')
        groups[key].append(index)
    if not parameters or any(not indices for indices in groups.values()):raise ValueError('Whole coarse/action/prior required')
    return parameters,groups


def geometry(losses,parameters,groups):
    if set(losses)!=set(d.OBJECTIVES) or not parameters:raise ValueError('Three complete scalar objectives required')
    vectors={}
    for i,name in enumerate(d.OBJECTIVES):
        loss=losses[name]
        if loss.ndim or not torch.isfinite(loss):raise FloatingPointError('Finite scalar source objective required')
        values=torch.autograd.grad(loss,parameters,retain_graph=i<len(d.OBJECTIVES)-1,allow_unused=True)
        parts=[torch.zeros_like(p).flatten() if v is None else v.detach().flatten() for p,v in zip(parameters,values)]
        if any(not torch.isfinite(v).all() for v in parts):raise FloatingPointError('Nonfinite source objective gradient')
        vectors[name]={group:torch.cat([parts[j] for j in indices]).double() for group,indices in groups.items()}
    result={}
    for group in groups:
        norms={name:float(torch.linalg.vector_norm(vectors[name][group])) for name in d.OBJECTIVES};pairs={}
        for first,second in d.PAIRS:
            a,b=norms[first],norms[second];dot=float(torch.dot(vectors[first][group],vectors[second][group]))
            valid=a>d.NORM_FLOOR and b>d.NORM_FLOOR
            pairs[first+'_vs_'+second]=dict(valid=valid,cosine=max(-1.,min(1.,dot/(a*b))) if valid else None,
                dot=dot,norm_ratio=b/a if a>d.NORM_FLOOR else None)
        result[group]=dict(norms=norms,pairs=pairs)
    return result


def measure_batch(model,received,clean,teacher):
    parameters,groups=parameter_groups(model);flags=[p.requires_grad for p in parameters]
    if any(p.grad is not None for p in parameters) or any(m.training for m in model.modules()):
        raise ValueError('Evaluation model with empty persistent gradients required')
    try:
        for p in parameters:p.requires_grad_(True)
        with torch.enable_grad():
            outputs=model.stage_outputs(received.detach())[0];identity=model(clean.detach())
            _,terms=objective(outputs,[v.detach() for v in teacher],clean.detach(),20,'moment_clean',identity)
            losses=dict(base=terms['base_loss'].mean(),moment_clean=terms['clean_moment_mse'].mean(),moment_teacher=terms['teacher_moment_mse'].mean())
            values={name:float(v.detach()) for name,v in losses.items()}
            measured=geometry(losses,parameters,groups)
        return dict(losses=values,geometry=measured)
    finally:
        for p,flag in zip(parameters,flags):p.requires_grad_(flag)


def _statistics(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return dict(mean=None,median=None,p90=None,p95=None,minimum=None,maximum=None)
    if not np.isfinite(values).all():raise ValueError('Finite source summaries required')
    return dict(mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)),minimum=float(values.min()),maximum=float(values.max()))


def summarize_batches(records):
    if len(records)!=16:raise ValueError('All sixteen fixed mixed-role batches required')
    rows=[]
    for name in d.OBJECTIVES:
        rows.append(dict(kind='loss',group='whole',metric=name,count=16,valid_count=16,invalid_count=0,negative_fraction=None,**_statistics([r['losses'][name] for r in records])))
    for group in d.GROUPS:
        for name in d.OBJECTIVES:
            rows.append(dict(kind='norm',group=group,metric=name,count=16,valid_count=16,invalid_count=0,negative_fraction=None,**_statistics([r['geometry'][group]['norms'][name] for r in records])))
        for first,second in d.PAIRS:
            name=first+'_vs_'+second;entries=[r['geometry'][group]['pairs'][name] for r in records]
            values=[v['cosine'] for v in entries if v['valid']]
            if any(v['cosine'] is not None for v in entries if not v['valid']):raise ValueError('Undefined gradient cosine must remain null')
            ratios=[v['norm_ratio'] for v in entries if v['norm_ratio'] is not None]
            rows.append(dict(kind='cosine',group=group,metric=name,count=16,valid_count=len(values),invalid_count=16-len(values),
                negative_fraction=sum(v<0 for v in values)/len(values) if values else None,
                norm_ratio_valid_count=len(ratios),norm_ratio_mean=float(np.mean(ratios)) if ratios else None,**_statistics(values)))
    if len(rows)!=27:raise ValueError('Complete loss,norm and pair matrix required')
    return rows
