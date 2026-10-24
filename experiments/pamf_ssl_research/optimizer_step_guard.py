"""Guard the actual extra AdamW parameter motion while retaining its native moments."""
from contextlib import contextmanager
from contextvars import ContextVar
import gzip
import json
import math
from pathlib import Path
import time
import torch
from experiments.cric_dura_comparison import f_optimizer
from .optimizer_reference import labeled_step
from .parameter_step_projection import project_parameters

_SCOPE=ContextVar('pamf_parameter_step_guard',default=None)


class OptimizerStepGuard:
    def __init__(self,destination=None):
        self.destination=Path(destination) if destination is not None else None
        self.pending=None;self.reference=None;self.optimizer=None;self.hook=None;self.stream=None
        self.calls=0;self.requests=0;self.fallback_groups=0;self.conflicts_removed=0
        self.minimum_retention=1.;self.seconds=0.;self.last_details={}

    def capture(self,model,named,full,clean,physical):
        scope=_SCOPE.get()
        if scope is None or scope['pending'] is not None or self.pending is not None:
            raise ValueError('One active step scope and no unconsumed anchors required')
        if any(len(v)!=len(named) for v in (full,clean,physical)):
            raise ValueError('Three complete original supervised gradients required')
        anchors=tuple(tuple(None if g is None else g.detach().clone() for g in v) for v in (full,clean,physical))
        self.pending=(model,list(named),anchors);self.requests+=1
        scope['pending']=self
        if self not in scope['guards']:scope['guards'].append(self)

    def before_clip(self,optimizer,parameters,max_norm,clip_impl):
        if self.pending is None or self.reference is not None or float(max_norm)!=5.:
            raise ValueError('One unconsumed current L reference at original physical clip5 required')
        model,named,anchors=self.pending
        actual=[p for g in getattr(optimizer,'param_groups',[]) for p in g['params']]
        if ({id(p) for p in parameters}!={id(p) for _,p in named}
                or len(actual)!=len(named) or {id(p) for p in actual}!={id(p) for _,p in named}):
            raise ValueError('The actual original model optimizer and all its parameters required')
        if self.optimizer is not None and self.optimizer is not optimizer:
            raise ValueError('One native segment optimizer per objective required')
        self.started=time.perf_counter()
        self.reference,self.reference_meta=labeled_step(model,named,optimizer,anchors[0],clip_impl=clip_impl)
        if self.hook is None:
            self.optimizer=optimizer
            self.hook=optimizer.register_step_post_hook(self.after_step)

    @torch.no_grad()
    def after_step(self,optimizer,args,kwargs):
        if self.pending is None or self.reference is None or optimizer is not self.optimizer:
            raise ValueError('Actual joint update lacks the matching L-only reference')
        model,named,anchors=self.pending
        raw=tuple(p.detach() for _,p in named)
        values,details=project_parameters(named,self.reference,raw,*anchors)
        for (_,parameter),value in zip(named,values):parameter.copy_(value)
        # The real joint AdamW state is retained; projection changes only parameters.
        self.calls+=1;self.last_details=details
        raw_norm=math.sqrt(sum(d['raw_increment_norm']**2 for d in details.values()))
        applied=math.sqrt(sum(d['applied_increment_norm']**2 for d in details.values()))
        retention=applied/raw_norm if raw_norm else 1.
        self.minimum_retention=min(self.minimum_retention,retention)
        fallbacks=sum(d['quantization_fallback'] for d in details.values());self.fallback_groups+=fallbacks
        removed=sum(before is not None and before<-1e-6 and (d['anchor_cosines_after'][anchor] is None
                    or d['anchor_cosines_after'][anchor]>=-1e-6)
                    for d in details.values() for anchor,before in d['anchor_cosines_before'].items())
        self.conflicts_removed+=removed
        seconds=time.perf_counter()-self.started;self.seconds+=seconds
        if self.destination is not None:
            if self.stream is None:self.stream=gzip.open(self.destination,'xt',encoding='utf-8',compresslevel=3)
            self.stream.write(json.dumps(dict(step=self.calls,groups=details,whole_increment_retention=retention,
                reference=self.reference_meta,seconds=seconds),allow_nan=False)+'\n');self.stream.flush()
        self.pending=None;self.reference=None
        scope=_SCOPE.get()
        if scope is None or scope['pending'] is not self:raise ValueError('Parameter step escaped its native scope')
        scope['pending']=None

    def state_dict(self):
        return dict(calls=self.calls,anchor_captures=self.requests,reference_optimizer_steps=self.calls,
                    reference_physical_clip_calls=self.calls,pending_anchors=self.pending is not None,
                    copied_existing_moments=True,live_joint_optimizer_state_retained=True,
                    reference_has_no_model_forward=True,extra_autograd_calls=0,extra_model_forwards=0,
                    fallback_groups=self.fallback_groups,removed_material_conflicts=self.conflicts_removed,
                    minimum_whole_increment_retention=self.minimum_retention,seconds=self.seconds,
                    U_truth_reads=0,target_reads=0,new_head=False,new_inference_parameters=0)

    def close(self):
        if self.hook is not None:self.hook.remove();self.hook=None
        if self.stream is not None:self.stream.close();self.stream=None


@contextmanager
def installed():
    if _SCOPE.get() is not None:raise ValueError('No nested parameter step scope')
    scope=dict(pending=None,guards=[]);token=_SCOPE.set(scope);original=f_optimizer.clip_physical
    def clipping(opt,parameters,max_norm,*args,**kwargs):
        parameters=list(parameters);guard=scope['pending']
        if guard is not None:
            # native_scope passes the original clipping implementation explicitly.
            impl=kwargs.get('clip_impl',torch.nn.utils.clip_grad_norm_)
            guard.before_clip(opt,parameters,max_norm,impl)
        return original(opt,parameters,max_norm,*args,**kwargs)
    f_optimizer.clip_physical=clipping
    try:
        yield scope
        if scope['pending'] is not None:raise ValueError('Unconsumed parameter step at native segment end')
    finally:
        for guard in scope['guards']:guard.close()
        f_optimizer.clip_physical=original;_SCOPE.reset(token)
