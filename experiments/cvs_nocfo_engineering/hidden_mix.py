"""One native map per active step; source-label feature/target interpolation."""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cvs_e20_cross_style.style import _paths
from .update import rng_state,set_rng

def plan(labels,seed,step,device):
    if labels.ndim!=1 or labels.dtype!=torch.long or not bool(((labels>=0)&(labels<6)).all()):raise ValueError('Visible source labels required')
    y=labels.detach().cpu();gen=torch.Generator().manual_seed(seed+step*104729+8107)
    donors=[];eligible=[]
    for i,tx in enumerate(y):
        pool=torch.where(y!=tx)[0]
        donor=int(pool[int(torch.randint(len(pool),(1,),generator=gen))]) if len(pool) else i
        donors.append(donor);eligible.append(bool(len(pool)))
    beta=np.random.default_rng(seed+step*65537+2701).beta(2.,2.,size=len(y)).tolist()
    lam=[max(v,1-v) if ok else 1. for v,ok in zip(beta,eligible)]
    return torch.tensor(donors,dtype=torch.long,device=device),torch.tensor(lam,dtype=torch.float32,device=device),eligible

@contextmanager
def capture(model,level,donors,lam):
    _,paths=_paths(model.base);module=dict(model.base.named_modules())[paths[level]];seen=[]
    def hook(module,args,h):
        if h.ndim!=3 or h.shape[0]!=len(donors) or seen:raise ValueError('Invalid native hidden-mix map')
        seen.append(list(h.shape));w=lam.to(h.dtype).reshape(-1,1,1)
        return w*h+(1-w)*h[donors]
    handle=module.register_forward_hook(hook)
    try:
        yield seen
        if len(seen)!=1:raise ValueError('Selected hidden-mix layer not executed exactly once')
    finally:handle.remove()

def forward(c,model,x,labels,step,clean_rng):
    if step<=444 or not c['engineering']['hidden_mix']:raise ValueError('Unregistered hidden mix forward')
    level=c['engineering']['mix_levels'][(step-445)%2];donors,lam,eligible=plan(labels,c['model_seed'],step,x.device)
    q=F.one_hot(labels,6).to(x.dtype);target=lam[:,None]*q+(1-lam[:,None])*q[donors]
    end=rng_state()
    try:
        set_rng(clean_rng)
        with capture(model,level,donors,lam) as shapes:logits=model(x)
    finally:set_rng(end)
    loss=-(target*logits.log_softmax(1)).sum(1).mean()
    changed=int(sum(eligible))
    return loss,dict(source_role='L_s',U_truth=False,V_access=False,target_access=False,
        level=level,native_shape=shapes[0],donors=donors.tolist(),lambda_values=lam.tolist(),different_TX_verified=all(int(labels[i])!=int(labels[j]) for i,j in enumerate(donors.tolist()) if eligible[i]),
        eligible_packets=changed,changed_packets=changed,skipped_packets=len(labels)-changed,
        extra_forward_calls=1,soft_target_sum_error=float((target.sum(1)-1).abs().max()),
        same_lambda_feature_target=True,one_layer_only=True,main_rng_restored=True,
        dropout_clean_replay=True,loss=float(loss.detach()))
