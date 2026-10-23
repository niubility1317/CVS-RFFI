"""Project only U onto full-T1-L/raw-CE halfspaces; default model unchanged."""
import math
import torch
from .objective import Objective
from .u_gradprotect_objective import parameter_group,_ExplicitGradient


def _coefficients(d0,d1,cosine,u_norm):
    tolerance=1e-10*max(1.,u_norm)
    gram=torch.tensor([[1.,cosine],[cosine,1.]],dtype=torch.float64)
    both=torch.linalg.pinv(gram,rtol=1e-12,atol=0.)@torch.tensor([-d0,-d1],dtype=torch.float64)
    candidates=[('none',0.,0.),('full',-d0,0.),('raw',0.,-d1),('both',float(both[0]),float(both[1]))]
    feasible=[]
    for active,a,b in candidates:
        if d0+a+cosine*b>=-tolerance and d1+cosine*a+b>=-tolerance:
            feasible.append((max(0.,a*a+b*b+2*cosine*a*b),active,a,b))
    # Zero is always feasible; retain the orthogonal U component when possible.
    feasible.append((u_norm*u_norm,'zero',None,None))
    return min(feasible,key=lambda item:item[0])[1:]


@torch.no_grad()
def dual_projected_gradients(named,full,raw,unlabeled):
    if not len(named)==len(full)==len(raw)==len(unlabeled):
        raise ValueError('Dual gradient/parameter lengths differ')
    combined=[None]*len(named);details={}
    for group in ('t2','cls_head','other'):
        indices=[i for i,(name,_) in enumerate(named) if parameter_group(name)==group
            and any(values[i] is not None for values in (full,raw,unlabeled))]
        tensors=[]
        for values in (full,raw,unlabeled):
            tensors.append([torch.zeros_like(named[i][1]) if values[i] is None else values[i].detach() for i in indices])
        gl,gr,gu=tensors
        if any(not bool(torch.isfinite(g).all()) for values in tensors for g in values):
            raise ValueError('Dual source gradients must be finite')
        def dot(left,right):
            return sum(float((g.double()*h.double()).sum()) for g,h in zip(left,right))
        ll,rr,uu=dot(gl,gl),dot(gr,gr),dot(gu,gu)
        dl,dr=dot(gl,gu),dot(gr,gu)
        nl,nr,nu=math.sqrt(ll),math.sqrt(rr),math.sqrt(uu)
        tiny=ll<=1e-12 or rr<=1e-12
        if tiny:
            active='tiny';projected=[torch.zeros_like(g) for g in gu]
        else:
            cosine=max(-1.,min(1.,dot(gl,gr)/(nl*nr)))
            active,a,b=_coefficients(dl/nl,dr/nr,cosine,nu)
            projected=([torch.zeros_like(g) for g in gu] if active=='zero' else
                [u+(a/nl)*l+(b/nr)*r for l,r,u in zip(gl,gr,gu)])
        projected_norm=math.sqrt(dot(projected,projected))
        fallback=(dot(gl,projected)<-1e-6*max(1.,nl*projected_norm)
                  or dot(gr,projected)<-1e-6*max(1.,nr*projected_norm))
        if fallback:
            projected=[torch.zeros_like(g) for g in gu];projected_norm=0.
        scale=1. if projected_norm==0 else min(1.,nl/projected_norm)
        applied=[g*scale for g in projected]
        for i,l,u in zip(indices,gl,applied):
            combined[i]=l+u
        details[group]=dict(full_L_norm=nl,raw_L_norm=nr,U_norm=nu,
            applied_U_norm=math.sqrt(dot(applied,applied)),full_dot_before=dl,raw_dot_before=dr,
            full_dot_after=dot(gl,applied),raw_dot_after=dot(gr,applied),active_set=active,
            full_conflict=dl<0.,raw_conflict=dr<0.,capped=scale<1.,tiny_anchor=tiny,numerical_fallback=fallback)
    return tuple(combined),details


def dual_protected_loss(full,raw,unlabeled,named):
    parameters=tuple(p for _,p in named)
    gl=torch.autograd.grad(full,parameters,retain_graph=True,allow_unused=True)
    gr=torch.autograd.grad(raw,parameters,retain_graph=True,allow_unused=True)
    gu=torch.autograd.grad(unlabeled,parameters,retain_graph=True,allow_unused=True)
    gradients,details=dual_projected_gradients(named,gl,gr,gu)
    loss=_ExplicitGradient.apply((full+unlabeled).detach(),gradients,*parameters)
    return loss,details,gradients


class DualAnchorObjective(Objective):
    def __init__(self,student,config):
        super().__init__(student,config)
        self.pending_gradient_loss=self.pending_raw_anchor=None
        self.u_dualanchor_calls=self.u_dualanchor_raw_calls=0
        self.u_dualanchor_active_sets={g:{k:0 for k in ('none','full','raw','both','zero','tiny')}
                                      for g in ('t2','cls_head','other')}
        self.u_dualanchor_fallbacks={g:0 for g in self.u_dualanchor_active_sets}

    def observe_raw(self,value):
        if self.pending_raw_anchor is not None or self.current_epoch<31 or not value.requires_grad or value.ndim:
            raise ValueError('One existing differentiable raw-L CE anchor per update required')
        self.pending_raw_anchor=value;self.u_dualanchor_raw_calls+=1

    def loss(self,student,device):
        if self.pending_gradient_loss is not None or self.current_epoch<31:
            raise ValueError('One hidden-label U graph per dual-anchor update required')
        self.pending_gradient_loss=super().loss(student,device)
        return self.pending_gradient_loss.detach()*0.

    def guide(self,full,student):
        if self.pending_raw_anchor is None:
            raise ValueError('Current existing raw-L CE graph required')
        if self.pending_gradient_loss is None:
            raise ValueError('Current hidden-label U graph required')
        raw,unlabeled=self.pending_raw_anchor,self.pending_gradient_loss
        self.pending_raw_anchor=self.pending_gradient_loss=None
        named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
        loss,detail,gradients=dual_protected_loss(full,raw,unlabeled,named)
        self.u_dualanchor_calls+=1
        self.metrics.update(SSL_U_DA_calls=self.u_dualanchor_calls,SSL_U_DA_raw_calls=self.u_dualanchor_raw_calls,
                            SSL_U_DA_raw_CE=float(raw.detach()))
        for group,values in detail.items():
            self.u_dualanchor_active_sets[group][values['active_set']]+=1
            self.u_dualanchor_fallbacks[group]+=int(values['numerical_fallback'])
            for key,value in values.items():
                if key!='active_set':self.metrics['SSL_U_DA_'+group+'_'+key]=float(value)
            for active,count in self.u_dualanchor_active_sets[group].items():
                self.metrics['SSL_U_DA_'+group+'_active_'+active]=count
            self.metrics['SSL_U_DA_'+group+'_fallbacks']=self.u_dualanchor_fallbacks[group]
        if self.c.get('u_dualanchor_smoke'):
            self._smoke_expected={n:g for (n,_),g in zip(named,gradients)}
            self._smoke_detail=detail;self._smoke_raw_value=float(raw.detach())
        return loss

    def state_dict(self):
        state=super().state_dict()
        state.update(u_dualanchor=self.c['u_dualanchor'],u_dualanchor_calls=self.u_dualanchor_calls,
            u_dualanchor_raw_calls=self.u_dualanchor_raw_calls,u_dualanchor_active_sets=self.u_dualanchor_active_sets,
            u_dualanchor_fallbacks=self.u_dualanchor_fallbacks)
        return state

    def load_state_dict(self,state):
        if state.get('u_dualanchor') and state['u_dualanchor']!=self.c['u_dualanchor']:
            raise ValueError('Dual anchor state belongs to another recipe')
        super().load_state_dict(state)
        self.u_dualanchor_calls=state.get('u_dualanchor_calls',0)
        self.u_dualanchor_raw_calls=state.get('u_dualanchor_raw_calls',0)
        self.u_dualanchor_active_sets={g:dict(values) for g,values in state.get('u_dualanchor_active_sets',self.u_dualanchor_active_sets).items()}
        self.u_dualanchor_fallbacks=dict(state.get('u_dualanchor_fallbacks',self.u_dualanchor_fallbacks))
