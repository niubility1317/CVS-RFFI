"""Project each existing U-view gradient, then allocate equal bounded norms."""
import math
import torch
from .objective import Objective
from .u_gradprotect_objective import parameter_group,_ExplicitGradient
from .u_dualanchor_objective import _coefficients

VIEWS=('ordinary','physical')


@torch.no_grad()
def balanced_gradients(named,full,raw,ordinary,physical):
    if not len(named)==len(full)==len(raw)==len(ordinary)==len(physical):
        raise ValueError('View gradient/parameter lengths differ')
    combined=[None]*len(named);details={}
    for group in ('t2','cls_head','other'):
        indices=[i for i,(name,_) in enumerate(named) if parameter_group(name)==group
            and any(values[i] is not None for values in (full,raw,ordinary,physical))]
        tensors=[[torch.zeros_like(named[i][1]) if values[i] is None else values[i].detach() for i in indices]
                 for values in (full,raw,ordinary,physical)]
        if any(not bool(torch.isfinite(g).all()) for values in tensors for g in values):
            raise ValueError('View source gradients must be finite')
        gl,gr,go,gp=tensors
        def dot(left,right):return sum(float((a.double()*b.double()).sum()) for a,b in zip(left,right))
        ll,rr=dot(gl,gl),dot(gr,gr);nl,nr=math.sqrt(ll),math.sqrt(rr)
        tiny=ll<=1e-12 or rr<=1e-12
        cosine=0. if tiny else max(-1.,min(1.,dot(gl,gr)/(nl*nr)))
        group_detail=dict(full_L_norm=nl,raw_L_norm=nr,tiny_anchor=tiny);applied_views=[]
        for view,gu in zip(VIEWS,(go,gp)):
            uu=dot(gu,gu);nu=math.sqrt(uu);dl,dr=dot(gl,gu),dot(gr,gu)
            if tiny:
                active='tiny';projected=[torch.zeros_like(g) for g in gu]
            else:
                active,a,b=_coefficients(dl/nl,dr/nr,cosine,nu)
                projected=([torch.zeros_like(g) for g in gu] if active=='zero' else
                    [u+(a/nl)*l+(b/nr)*r for l,r,u in zip(gl,gr,gu)])
            norm2=dot(projected,projected);projected_norm=math.sqrt(norm2);tiny_projected=norm2<=1e-12
            scale=0. if tiny_projected else .5*nl/projected_norm
            applied=[g*scale for g in projected]
            applied_norm=math.sqrt(dot(applied,applied))
            fallback=(dot(gl,applied)<-1e-6*max(1.,nl*applied_norm)
                      or dot(gr,applied)<-1e-6*max(1.,nr*applied_norm))
            if fallback:
                applied=[torch.zeros_like(g) for g in gu];applied_norm=0.;scale=0.
            applied_views.append(applied)
            group_detail[view]=dict(U_norm=nu,projected_U_norm=projected_norm,applied_U_norm=applied_norm,scale=scale,
                full_dot_before=dl,raw_dot_before=dr,full_dot_after=dot(gl,applied),raw_dot_after=dot(gr,applied),
                active_set=active,tiny_projected=tiny_projected,numerical_fallback=fallback)
        for index,l,o,p in zip(indices,gl,*applied_views):combined[index]=l+o+p
        total=[o+p for o,p in zip(*applied_views)]
        group_detail.update(total_U_norm=math.sqrt(dot(total,total)),full_dot_after=dot(gl,total),raw_dot_after=dot(gr,total))
        details[group]=group_detail
    return tuple(combined),details


def balanced_loss(full,raw,ordinary,physical,named):
    parameters=tuple(p for _,p in named)
    gradients=[torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
               for loss in (full,raw,ordinary,physical)]
    prepared,detail=balanced_gradients(named,*gradients)
    loss=_ExplicitGradient.apply((full+ordinary+physical).detach(),prepared,*parameters)
    return loss,detail,prepared


class ViewBalancedObjective(Objective):
    def __init__(self,student,config):
        super().__init__(student,config)
        self.pending_raw_anchor=self.pending_views=self.pending_total_U=None
        self.pending_view_counts=None;self.u_viewbalance_calls=self.u_viewbalance_raw_calls=0
        self.u_viewbalance_view_seen={view:0 for view in VIEWS}
        self.u_viewbalance_upscaled={g:{view:0 for view in VIEWS} for g in ('t2','cls_head','other')}
        self.u_viewbalance_dropped={g:{view:0 for view in VIEWS} for g in self.u_viewbalance_upscaled}

    def observe_raw(self,value):
        if self.pending_raw_anchor is not None or self.current_epoch<31 or not value.requires_grad or value.ndim:
            raise ValueError('One existing differentiable raw-L CE anchor per update required')
        self.pending_raw_anchor=value;self.u_viewbalance_raw_calls+=1

    def unlabeled_view_contributions(self,per_record,mask,coefficient):
        if self.pending_views is not None or per_record.ndim!=1 or len(per_record)!=len(mask) or mask.dtype!=torch.bool:
            raise ValueError('One exact original U-view partition per update required')
        n=len(per_record)
        if not n:raise ValueError('Nonempty original hidden U batch required')
        self.pending_views=(coefficient*per_record[~mask].sum()/n,coefficient*per_record[mask].sum()/n)
        self.pending_view_counts=dict(ordinary=int((~mask).sum()),physical=int(mask.sum()))

    def loss(self,student,device):
        if self.pending_views is not None or self.pending_total_U is not None or self.current_epoch<31:
            raise ValueError('One current hidden-U view graph required')
        self.pending_total_U=super().loss(student,device)
        if self.pending_views is None or not torch.allclose(sum(self.pending_views),self.pending_total_U,atol=1e-7,rtol=1e-6):
            raise ValueError('Original weighted U scalar decomposition differs')
        return self.pending_total_U.detach()*0.

    def guide(self,full,student):
        if self.pending_raw_anchor is None:raise ValueError('Current existing raw-L CE graph required')
        if self.pending_views is None or self.pending_total_U is None:raise ValueError('Current original two U-view graphs required')
        raw=self.pending_raw_anchor;ordinary,physical=self.pending_views;counts=self.pending_view_counts
        self.pending_raw_anchor=self.pending_views=self.pending_total_U=self.pending_view_counts=None
        named=[(name,p) for name,p in student.named_parameters() if p.requires_grad]
        loss,detail,gradients=balanced_loss(full,raw,ordinary,physical,named)
        self.u_viewbalance_calls+=1
        for view in VIEWS:self.u_viewbalance_view_seen[view]+=counts[view]
        self.metrics.update(SSL_U_VB_calls=self.u_viewbalance_calls,SSL_U_VB_raw_calls=self.u_viewbalance_raw_calls,
            SSL_U_VB_raw_CE=float(raw.detach()),SSL_U_VB_ordinary_scalar=float(ordinary.detach()),
            SSL_U_VB_physical_scalar=float(physical.detach()),
            SSL_U_VB_ordinary_seen=self.u_viewbalance_view_seen['ordinary'],SSL_U_VB_physical_seen=self.u_viewbalance_view_seen['physical'])
        for group,values in detail.items():
            for key in ('full_L_norm','raw_L_norm','tiny_anchor','total_U_norm','full_dot_after','raw_dot_after'):
                self.metrics['SSL_U_VB_'+group+'_'+key]=float(values[key])
            for view in VIEWS:
                v=values[view];self.u_viewbalance_upscaled[group][view]+=int(v['scale']>1.)
                self.u_viewbalance_dropped[group][view]+=int(v['tiny_projected'] or v['numerical_fallback'])
                prefix='SSL_U_VB_'+group+'_'+view+'_'
                for key,value in v.items():
                    if key!='active_set':self.metrics[prefix+key]=float(value)
                self.metrics[prefix+'active_set']=('none','full','raw','both','zero','tiny').index(v['active_set'])
                self.metrics[prefix+'upscaled']=self.u_viewbalance_upscaled[group][view]
                self.metrics[prefix+'dropped']=self.u_viewbalance_dropped[group][view]
        if self.c.get('u_viewbalance_smoke'):
            self._smoke_expected={name:g for (name,_),g in zip(named,gradients)}
            self._smoke_detail=detail;self._smoke_raw_value=float(raw.detach());self._smoke_counts=counts
        return loss

    def state_dict(self):
        value=super().state_dict();value.update(u_viewbalance=self.c['u_viewbalance'],
            u_viewbalance_calls=self.u_viewbalance_calls,u_viewbalance_raw_calls=self.u_viewbalance_raw_calls,
            u_viewbalance_view_seen=self.u_viewbalance_view_seen,u_viewbalance_upscaled=self.u_viewbalance_upscaled,
            u_viewbalance_dropped=self.u_viewbalance_dropped)
        return value

    def load_state_dict(self,state):
        if state.get('u_viewbalance') and state['u_viewbalance']!=self.c['u_viewbalance']:
            raise ValueError('View balance state belongs to another recipe')
        super().load_state_dict(state)
        self.u_viewbalance_calls=state.get('u_viewbalance_calls',0);self.u_viewbalance_raw_calls=state.get('u_viewbalance_raw_calls',0)
        self.u_viewbalance_view_seen=dict(state.get('u_viewbalance_view_seen',self.u_viewbalance_view_seen))
        for name in ('u_viewbalance_upscaled','u_viewbalance_dropped'):
            setattr(self,name,{g:dict(v) for g,v in state.get(name,getattr(self,name)).items()})
