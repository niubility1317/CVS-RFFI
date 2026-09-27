"""Experimental exact-objective graph reuse; isolated head parameters only.

The helper is deliberately Core90-specific. The original full objective graph
retains FISHR, U, prototype and all non-adversarial terms. No online parameter
is mutated while that graph is alive.
"""
import copy
import torch
import torch.nn.functional as F
from .state import RNGState


class Core90ReusableGraph:
    def __init__(self, objective, ctx):
        self.objective, self.ctx = objective, ctx

    def __call__(self):
        self.calls=[]
        def before(module, inputs):
            self.calls.append([inputs[0], RNGState.capture(), None])
        def after(module, inputs, output):
            self.calls[-1][2]=output
        head=self.objective.model.adv_head
        pre=head.register_forward_pre_hook(before)
        post=head.register_forward_hook(after)
        try: self.loss=self.objective(self.ctx)
        finally: pre.remove(); post.remove()
        if not self.calls:
            raise NotImplementedError('reusable objective did not execute adversarial head')
        return self.loss

    def _adversarial_loss(self, logits):
        n=len(self.ctx.y); valid=self.ctx.domain>=0
        scale=self.ctx.weights['adv']
        if self.objective.args.game_head_scale=='separate_head_scale': scale=float(scale>0)
        return scale*F.cross_entropy(logits[:n][valid].float(),self.ctx.domain[valid]) if scale>0 and bool(valid.any()) else None

    def component_gradients(self, parameters, *, loss=None, logits=None, targets=None):
        """Read-only extra reverse passes, requested only on telemetry steps."""
        loss=self.loss if loss is None else loss
        logits=self.calls[0][2] if logits is None else logits
        active=[p for p in parameters if p.requires_grad]
        targets=active if targets is None else targets
        adv=self._adversarial_loss(logits)
        def gradient(value):
            rows=torch.autograd.grad(value,targets,retain_graph=True,allow_unused=True) if value is not None else [None]*len(targets)
            by_id={id(p):g for p,g in zip(active,rows)}
            return [None if by_id.get(id(p)) is None else by_id[id(p)].detach().clone() for p in parameters]
        return {'adv':gradient(adv),'nonadv':gradient(loss if adv is None else loss-adv),
                'backward_calls':1+int(adv is not None)}

    def corrector(self, first, solver):
        head=self.objective.model.adv_head
        # Clone only the head and its owned optimizer groups/moments. AdamW
        # grad=None retains native no-decay/no-step semantics.
        virtual=copy.deepcopy(head)
        online=list(head.parameters()); isolated=list(virtual.parameters())
        mapping={id(p):q for p,q in zip(online,isolated)}
        groups=[]
        for group in solver.optimizer.param_groups:
            params=[mapping[id(p)] for p in group['params'] if id(p) in mapping]
            if params: groups.append({**{k:copy.deepcopy(v) for k,v in group.items() if k!='params'},'params':params})
        opt=type(solver.optimizer)(groups)
        grads={id(p):g for p,g in zip(solver.parameters,first)}
        for p,q in zip(online,isolated):
            if p in solver.optimizer.state: opt.state[q]=copy.deepcopy(solver.optimizer.state[p])
            q.grad=None if grads[id(p)] is None else grads[id(p)].clone()
        if solver.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(isolated,solver.max_grad_norm,error_if_nonfinite=True)
        self.predictor_clipped=[None if id(p) not in mapping or mapping[id(p)].grad is None else mapping[id(p)].grad.detach().clone() for p in solver.parameters] if solver._measure_components else None
        opt.step()
        for q in isolated: q.grad=None
        current=RNGState.capture()
        features,rng,old_logits=self.calls[0]
        try:
            rng.restore()
            new_logits=virtual(features)
        finally: current.restore()
        n=len(self.ctx.y); valid=self.ctx.domain>=0
        scale=self.ctx.weights['adv']
        if self.objective.args.game_head_scale=='separate_head_scale': scale=float(scale>0)
        if scale>0 and bool(valid.any()):
            old=F.cross_entropy(old_logits[:n][valid].float(),self.ctx.domain[valid])
            new=F.cross_entropy(new_logits[:n][valid].float(),self.ctx.domain[valid])
            loss=self.loss-scale*old+scale*new
        else: loss=self.loss
        targets=[mapping.get(id(p),p) for p in solver.parameters if p.requires_grad]
        self.predictor_components=self.component_gradients(solver.parameters,loss=loss,logits=new_logits,targets=targets) if solver._measure_components else None
        gradients=torch.autograd.grad(loss,targets,allow_unused=True)
        by_id={id(p):g for p,g in zip([p for p in solver.parameters if p.requires_grad],gradients)}
        result=[None if by_id.get(id(p)) is None else by_id[id(p)].detach().clone() for p in solver.parameters]
        return float(loss.detach()),result,[mapping[id(p)].detach()-p.detach() if id(p) in mapping else torch.zeros_like(p) for p in solver.parameters]
