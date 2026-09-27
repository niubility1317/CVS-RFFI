"""Frozen-second-moment local metric anchored to a real AdamW predictor."""
import math
import torch
from .ir_types import ResponseMetric

def build_response_metric(optimizer_origin,predictor_record,h0,layout):
    if isinstance(optimizer_origin,torch.optim.Optimizer):
        if type(optimizer_origin) is not torch.optim.AdamW:raise ValueError('only AdamW is supported')
        origin=optimizer_origin.state_dict()
    else:origin=optimizer_origin
    if predictor_record.get('optimizer_class','AdamW')!='AdamW':raise ValueError('only AdamW is supported')
    c=predictor_record['clip_coefficient']
    if isinstance(c,bool) or not math.isfinite(float(c)) or not 0<=float(c)<=1:raise ValueError('invalid actual predictor clip coefficient')
    c=float(c);after=predictor_record['optimizer_after']
    slots=[];seen=set()
    for group in origin['param_groups']:
        for flag in ('amsgrad','maximize','fused','capturable','differentiable'):
            if group.get(flag,False):raise ValueError(f'unsupported AdamW flag: {flag}')
        if group.get('foreach') is True:raise ValueError('unverified foreach optimizer variant')
        for pid in group['params']:
            if pid not in seen:slots.append((pid,group));seen.add(pid)
    if len(slots)!=len(layout.names) or len(h0)!=len(layout.phi_indices):raise ValueError('optimizer/head layout mismatch')
    values=[];indices=[];offset=0;reference=next((g for g in h0 if g is not None),None)
    for index,g in zip(layout.phi_indices,h0):
        pid,group=slots[index];numel=math.prod(layout.shapes[index]);lr=float(group['lr'])
        if g is None:offset+=numel;continue
        if g.shape!=layout.shapes[index] or g.is_sparse or g.dtype!=torch.float32 or not torch.isfinite(g).all():raise ValueError('unsupported gradient dtype/layout/finite state')
        state=origin['state'].get(pid,{})
        for name in ('exp_avg','exp_avg_sq'):
            if name in state and (state[name].dtype!=torch.float32 or state[name].shape!=g.shape or not torch.isfinite(state[name]).all()):raise ValueError('unsupported optimizer state')
        raw_step=float(state.get('step',0))
        if raw_step<0 or not raw_step.is_integer():raise ValueError('invalid parameter-local step')
        k=int(raw_step)+1;beta1,beta2=group['betas'];eps=float(group['eps'])
        if not (0<=beta1<1 and 0<=beta2<1 and eps>=0 and math.isfinite(lr) and lr>=0):raise ValueError('invalid AdamW group')
        v0=state.get('exp_avg_sq',torch.zeros_like(g))
        # Match torch AdamW's addcmul order rather than squaring a pre-scaled expression.
        clipped=g.detach()*c
        v1=v0.detach().clone().mul_(beta2).addcmul_(clipped,clipped,value=1-beta2)
        actual=after['state'].get(pid,{})
        if 'exp_avg_sq' not in actual or not torch.allclose(v1,actual['exp_avg_sq'],atol=1e-8,rtol=2e-6) or float(actual.get('step',-1))!=k:
            raise ValueError('predictor second moment/step does not match actual virtual step')
        if lr>0 and c>0:
            r=lr*c*(1-beta1)/((1-beta1**k)*(torch.sqrt(v1/(1-beta2**k))+eps))
            if not torch.isfinite(r).all() or (r<=0).any():raise ValueError('invalid response metric')
            values.append(r.reshape(-1));indices.append(torch.arange(offset,offset+numel,device=g.device))
        offset+=numel
    if reference is None:
        for state in origin['state'].values():
            if 'exp_avg_sq' in state:reference=state['exp_avg_sq'];break
    reference=torch.empty(0,dtype=torch.float32) if reference is None else reference
    diagonal=torch.cat(values) if values else reference.new_empty(0)
    active=torch.cat(indices) if indices else torch.empty(0,device=reference.device,dtype=torch.long)
    return ResponseMetric(diagonal,diagonal.sqrt(),active,c)
