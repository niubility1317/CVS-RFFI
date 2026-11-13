"""Observe same-state AdamW update deltas without changing live training."""
from contextlib import contextmanager
import gzip
import json
import math
from pathlib import Path

import torch
from experiments.cric_dura_comparison import f_optimizer
from experiments.pamf_ssl_research.optimizer_reference import labeled_step
from .branch_guard import GROUPS, parameter_group


class UpdateProbe:
    def __init__(self):
        self.pending=None;self.reference=None;self.before=None;self.hook=None;self.stream=None;self.calls=0

    def capture(self,model,named,base_gradients,step,path):
        if self.pending is not None:raise ValueError('Previous shadow update not consumed')
        self.pending=(model,named,base_gradients,step,Path(path))

    def before_clip(self,opt,clip_impl):
        model,named,base,step,path=self.pending
        self.before=tuple(p.detach().clone() for _,p in named)
        # The disposable shadow uses this exact optimizer state, physical clip and AdamW policy.
        pending=self.pending;self.pending=None
        try:self.reference,_=labeled_step(model,named,opt,base,clip_impl=clip_impl)
        finally:self.pending=pending
        if self.hook is None:self.hook=opt.register_step_post_hook(self.after_step)

    @torch.no_grad()
    def after_step(self,opt,args,kwargs):
        if self.pending is None:return
        model,named,_,step,path=self.pending
        records={}
        for group in GROUPS:
            values=[(p.detach().double()-ref.double(),ref.double()-before.double())
                    for (name,p),ref,before in zip(named,self.reference,self.before) if parameter_group(name)==group]
            if values:
                delta=math.sqrt(sum(float(a.square().sum()) for a,_ in values))
                baseline=math.sqrt(sum(float(b.square().sum()) for _,b in values))
                dot=sum(float((a*b).sum()) for a,b in values)
                records[group]=dict(aux_update_delta_norm=delta,base_update_norm=baseline,
                                    delta_dot_base=dot,delta_base_cosine=dot/(delta*baseline) if delta and baseline else None)
        if self.stream is None:self.stream=gzip.open(path,'wt',encoding='utf8',compresslevel=1)
        self.stream.write(json.dumps(dict(step=step,groups=records,reference='same live parameters, AdamW moments and physical clipping'),allow_nan=False)+'\n')
        self.stream.flush();self.calls+=1
        self.pending=self.reference=self.before=None

    def close(self):
        if self.hook is not None:self.hook.remove()
        if self.stream is not None:self.stream.close()


@contextmanager
def installed():
    probe=UpdateProbe();original=f_optimizer.clip_physical
    def clipping(opt,parameters,max_norm,*args,**kwargs):
        if probe.pending is not None:
            probe.before_clip(opt,kwargs.get('clip_impl',torch.nn.utils.clip_grad_norm_))
        return original(opt,parameters,max_norm,*args,**kwargs)
    f_optimizer.clip_physical=clipping
    try:yield probe
    finally:
        probe.close();f_optimizer.clip_physical=original
