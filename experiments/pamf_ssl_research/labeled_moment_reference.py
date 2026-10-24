"""One current-coordinate L reference with independent native-segment L moments."""
from copy import deepcopy
import math
import torch
from experiments.cric_dura_comparison.f_optimizer import build,clip_physical
from .optimizer_reference import container


class LabeledMomentReference:
    def __init__(self):
        self.shadow=None;self.optimizer=None;self.live_optimizer=None;self.identity=None
        self.parameters=None;self.calls=0;self.closed=False

    @torch.no_grad()
    def step(self,model,named,optimizer,gradients,*,clip_impl):
        named=list(named);observed=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        policy=getattr(optimizer,'_sinc_unit_policy',{})
        actual=[p for group in optimizer.param_groups for p in group['params']]
        if (self.closed or not named or len(named)!=len(observed) or len(gradients)!=len(named)
                or any(n!=a or p is not q for (n,p),(a,q) in zip(named,observed))
                or any(p.dtype!=torch.float32 or p.device!=named[0][1].device for _,p in named)
                or policy.get('mode')!='CENTER_UNITS' or len(actual)!=len(named)
                or {id(p) for p in actual}!={id(p) for _,p in named}):
            raise ValueError('One open native CENTER_UNITS scope with exact ordered jointly trainable FP32 parameters required')
        for (_,p),g in zip(named,gradients):
            if g is not None and (g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device or not bool(torch.isfinite(g).all())):
                raise ValueError('Finite aligned original supervised gradients required')
        if (getattr(optimizer,'_sinc_unit_steps',None)!=self.calls
                or getattr(optimizer,'_sinc_physical_clip_calls',None)!=self.calls):
            raise ValueError('Exactly one independent reference per native segment step required')
        identity=(id(model),tuple((n,id(p)) for n,p in named))
        if self.optimizer is None:
            if self.calls or optimizer.state:raise ValueError('Fresh real optimizer segment required for fresh independent L moments')
            self.shadow=container(model,named);lookup=dict(self.shadow.named_parameters())
            self.parameters=[lookup[n] for n,_ in named]
            keys=('betas','eps','weight_decay','lr','amsgrad','maximize','foreach','capturable','differentiable','fused')
            options={key:optimizer.defaults[key] for key in keys if key in optimizer.defaults}
            options.update(lr=float(optimizer.param_groups[0]['lr']),eps=policy['base_eps'],weight_decay=policy['base_weight_decay'])
            self.optimizer=build(self.shadow,self.parameters,'CENTER_UNITS',**options)
            if self.optimizer.state:raise ValueError('Independent reference moments must start empty')
            self.live_optimizer=optimizer;self.identity=identity
        elif self.live_optimizer is not optimizer or self.identity!=identity:
            raise ValueError('The same model,parameters and native segment optimizer required')
        if len(self.optimizer.param_groups)!=len(optimizer.param_groups):raise ValueError('Matching native optimizer groups required')
        # Copy scalar/group options only; never load or copy the real joint state.
        for reference,live in zip(self.optimizer.param_groups,optimizer.param_groups):
            if len(reference['params'])!=len(live['params']):raise ValueError('Matching native parameter group sizes required')
            for key,value in live.items():
                if key!='params':reference[key]=deepcopy(value)
        for key in ('base_eps','base_weight_decay'):
            value=float(policy[key])
            if not math.isfinite(value) or value<0 or key=='base_eps' and value==0:raise ValueError('Finite native optimizer policy required')
            self.optimizer._sinc_unit_policy[key]=value
        self.optimizer.zero_grad(set_to_none=True)
        for (_,live),reference,g in zip(named,self.parameters,gradients):
            reference.copy_(live)
            reference.grad=None if g is None else g.detach().clone()
        norm=clip_physical(self.optimizer,self.parameters,5.,clip_impl=clip_impl)
        if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite independent supervised physical gradient norm')
        self.optimizer.step();self.calls+=1
        if (self.optimizer._sinc_unit_steps!=self.calls or self.optimizer._sinc_physical_clip_calls!=self.calls):
            raise ValueError('Exactly one actual physical reference step required')
        values=tuple(p.detach().clone() for p in self.parameters)
        if any(not bool(torch.isfinite(p).all()) for p in values):raise ValueError('Nonfinite independent L reference parameters')
        return values,dict(reference_optimizer_steps=1,reference_physical_clip_calls=1,copied_existing_moments=False,
            independent_labeled_moments=True,reference_moment_steps=self.calls,
            reference_moment_origin='fresh_per_native_segment_L_only',reference_coordinates='current_live_parameters',
            live_optimizer_updates=0,model_forward_calls=0,physical_grad_norm_before_clip=float(norm))

    def close(self):
        self.shadow=self.optimizer=self.live_optimizer=self.parameters=self.identity=None;self.closed=True
