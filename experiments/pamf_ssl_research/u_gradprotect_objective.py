"""One-sided, L-anchored U gradients; default network and scalar losses retained."""
import math
import torch
from .objective import Objective


def parameter_group(name):
    return 't2' if '.t2.' in name else 'cls_head' if '.cls_head.' in name else 'other'


@torch.no_grad()
def projected_gradients(named, labeled, unlabeled, *, max_u_ratio=1.):
    if not isinstance(max_u_ratio,(int,float)) or not math.isfinite(max_u_ratio) or max_u_ratio<=0:
        raise ValueError('Finite positive U/L norm ratio required')
    if not len(named) == len(labeled) == len(unlabeled):
        raise ValueError('Gradient/parameter lengths differ')
    combined = [None] * len(named)
    details = {}
    for group in ('t2', 'cls_head', 'other'):
        indices = [i for i, (name, _) in enumerate(named) if parameter_group(name) == group
                   and (labeled[i] is not None or unlabeled[i] is not None)]
        if not indices:
            details[group] = dict(L_norm=0.,U_norm=0.,applied_U_norm=0.,dot_before=0.,dot_after=0.,
                                  conflict=False,capped=False,tiny_L=True)
            continue
        gl = [torch.zeros_like(named[i][1]) if labeled[i] is None else labeled[i].detach() for i in indices]
        gu = [torch.zeros_like(named[i][1]) if unlabeled[i] is None else unlabeled[i].detach() for i in indices]
        if any(not bool(torch.isfinite(g).all()) for g in gl + gu):
            raise ValueError('Source gradients must be finite')
        # Float64 reductions avoid cancellation in the group projection coefficient.
        ll = sum(g.double().square().sum() for g in gl)
        uu = sum(g.double().square().sum() for g in gu)
        dot = sum((g.double() * h.double()).sum() for g, h in zip(gl, gu))
        tiny = bool(ll <= 1e-12)
        conflict = bool(dot < 0.)
        if tiny:
            projected = [torch.zeros_like(g) for g in gu]
        else:
            coefficient = dot.clamp_max(0.) / ll
            projected = [h - coefficient.to(g.dtype) * g for g, h in zip(gl, gu)]
        norm2 = sum(g.double().square().sum() for g in projected)
        scale = torch.ones_like(ll) if bool(norm2 == 0.) else (max_u_ratio*(ll / norm2).sqrt()).clamp_max(1.)
        applied = [g * scale.to(g.dtype) for g in projected]
        for i, g, h in zip(indices, gl, applied):
            combined[i] = g + h
        details[group] = dict(L_norm=float(ll.sqrt()),U_norm=float(uu.sqrt()),
            applied_U_norm=float(norm2.sqrt() * scale),dot_before=float(dot),
            dot_after=float(sum((g.double()*h.double()).sum() for g,h in zip(gl,applied))),
            conflict=conflict,capped=bool(scale < 1.),tiny_L=tiny)
    return tuple(combined), details


class _ExplicitGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, gradients, *parameters):
        ctx.gradients = gradients
        return value.clone()

    @staticmethod
    def backward(ctx, scale):
        return (None, None, *(None if g is None else g * scale for g in ctx.gradients))


def protected_loss(labeled, unlabeled, named, *, max_u_ratio=1.):
    parameters = tuple(p for _, p in named)
    gl = torch.autograd.grad(labeled, parameters, retain_graph=True, allow_unused=True)
    gu = torch.autograd.grad(unlabeled, parameters, retain_graph=True, allow_unused=True)
    gradients, details = projected_gradients(named, gl, gu, max_u_ratio=max_u_ratio)
    value = _ExplicitGradient.apply((labeled + unlabeled).detach(), gradients, *parameters)
    return value, details, gradients


class GradientProtectedObjective(Objective):
    def __init__(self, student, config):
        super().__init__(student, config)
        self.pending_gradient_loss = None
        self.u_gradprotect_calls = 0
        self.u_gradprotect_conflicts = {g: 0 for g in ('t2','cls_head','other')}
        self.u_gradprotect_caps = dict(self.u_gradprotect_conflicts)

    def loss(self, student, device):
        if self.pending_gradient_loss is not None or self.current_epoch < 31:
            raise ValueError('One U gradient loss per E31+ labeled update required')
        self.pending_gradient_loss = super().loss(student, device)
        # The original T1 hook now builds its complete L graph without U gradients.
        return self.pending_gradient_loss.detach() * 0.

    def guide(self, labeled, student):
        if self.pending_gradient_loss is None:
            raise ValueError('Current hidden-label U loss required')
        unlabeled, self.pending_gradient_loss = self.pending_gradient_loss, None
        named = [(n,p) for n,p in student.named_parameters() if p.requires_grad]
        loss, detail, gradients = protected_loss(labeled, unlabeled, named)
        self.u_gradprotect_calls += 1
        self.metrics['SSL_U_GP_calls'] = self.u_gradprotect_calls
        for group, values in detail.items():
            self.u_gradprotect_conflicts[group] += int(values['conflict'])
            self.u_gradprotect_caps[group] += int(values['capped'])
            for key, value in values.items():
                self.metrics['SSL_U_GP_' + group + '_' + key] = float(value)
            self.metrics['SSL_U_GP_' + group + '_conflicts'] = self.u_gradprotect_conflicts[group]
            self.metrics['SSL_U_GP_' + group + '_caps'] = self.u_gradprotect_caps[group]
        if self.c.get('u_gradprotect_smoke'):
            self._smoke_expected = {n:g for (n,_),g in zip(named,gradients)}
            self._smoke_detail = detail
        return loss

    def state_dict(self):
        value = super().state_dict()
        value.update(u_gradprotect=self.c['u_gradprotect'],u_gradprotect_calls=self.u_gradprotect_calls,
                     u_gradprotect_conflicts=self.u_gradprotect_conflicts,u_gradprotect_caps=self.u_gradprotect_caps)
        return value

    def load_state_dict(self, state):
        if state.get('u_gradprotect') and state['u_gradprotect'] != self.c['u_gradprotect']:
            raise ValueError('Gradient state belongs to another recipe')
        super().load_state_dict(state)
        self.u_gradprotect_calls = state.get('u_gradprotect_calls',0)
        self.u_gradprotect_conflicts = dict(state.get('u_gradprotect_conflicts',self.u_gradprotect_conflicts))
        self.u_gradprotect_caps = dict(state.get('u_gradprotect_caps',self.u_gradprotect_caps))
