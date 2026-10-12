"""Real-coordinate geometry and native AdamW on independent parameter copies."""
from copy import deepcopy
import math
import torch

def add(a,b,scale=1.):
    return tuple(None if x is None and y is None else y*scale if x is None else x.clone() if y is None else x+y*scale for x,y in zip(a,b))

def geometry(a,b,indices):
    aa=bb=dot=0.
    for i in indices:
        x,y=a[i],b[i]
        if x is not None:aa+=float(x.detach().abs().to(torch.float64).square().sum())
        if y is not None:bb+=float(y.detach().abs().to(torch.float64).square().sum())
        if x is not None and y is not None:
            dtype=torch.complex128 if x.is_complex() or y.is_complex() else torch.float64
            dot+=float((x.detach().to(dtype).conj()*y.detach().to(dtype)).real.sum())
    na,nb=math.sqrt(aa),math.sqrt(bb)
    if not all(math.isfinite(v) for v in (na,nb,dot)):raise FloatingPointError('Nonfinite gradient/displacement geometry')
    cosine=dot/(na*nb) if na and nb else None
    if cosine is not None and not -1.000001<=cosine<=1.000001:raise ValueError('Invalid real-coordinate cosine')
    return dict(norm_a=na,norm_b=nb,dot=dot,cosine=max(-1.,min(1.,cosine)) if cosine is not None else None,
        norm_ratio_b_over_a=nb/na if na else None)

def clone_step(named,groups,saved_state,gradients,mode,weight_decay):
    """Never assign .grad or optimizer state to the checkpoint model itself."""
    if mode not in ('saved_final','reset_peak'):raise ValueError('Unregistered optimizer mode')
    if len(gradients)!=len(named):raise ValueError('Gradient/parameter length differs')
    originals={id(p):i for i,(_,p) in enumerate(named)}
    members=[id(p) for g in groups for p in g['params']]
    if len(members)!=len(set(members)) or set(members)!=set(originals):raise ValueError('Optimizer must cover each base parameter exactly once')
    copies=[torch.nn.Parameter(p.detach().clone(),requires_grad=True) for _,p in named]
    pg=[dict(**{k:deepcopy(v) for k,v in group.items() if k!='params'},params=[copies[originals[id(p)]] for p in group['params']]) for group in groups]
    optimizer=torch.optim.AdamW(pg,weight_decay=weight_decay)
    if mode=='saved_final':optimizer.load_state_dict(deepcopy(saved_state))
    for p,g in zip(copies,gradients):
        if g is not None:
            if g.shape!=p.shape or g.dtype!=p.dtype or not bool(torch.isfinite(g).all()):raise ValueError('Invalid assigned native gradient')
            p.grad=g.detach().clone()
    before=[p.detach().clone() for p in copies]
    optimizer.step()
    deltas=tuple(p.detach()-old for p,old in zip(copies,before))
    steps=[float(s['step']) for s in optimizer.state.values() if 'step' in s]
    info=dict(optimizer='torch.optim.AdamW',mode=mode,groups=[{k:g[k] for k in ('lr','betas','eps','weight_decay','amsgrad')} for g in optimizer.param_groups],
        native_complex_parameters=sum(p.is_complex() for p in copies),gradient_present=sum(g is not None for g in gradients),
        stored_moments_used=mode=='saved_final',new_checkpoint_saved=False,
        state_step_min=min(steps) if steps else None,state_step_max=max(steps) if steps else None)
    return deltas,info
