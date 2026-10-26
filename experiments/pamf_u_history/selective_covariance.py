"""Same-packet correlation consistency; training-L-only frozen mask selection.

This is an RF research variant, not RobustNet's covariance-to-zero objective.
No matrix inverse, identity head, whitening transform or inference branch.
"""
import math
import torch


class InsufficientMaskEvidence(ValueError):
    def __init__(self,diagnostic):
        super().__init__('Insufficient high-sensitivity/low-TX-association edges;do not relax silently')
        self.diagnostic=diagnostic


def correlation(h,ridge=1e-3):
    if (h.ndim!=3 or h.shape[1]<2 or h.shape[2]<2 or not h.is_floating_point()
            or not torch.isfinite(h).all() or not math.isfinite(ridge) or ridge<=0):
        raise ValueError('Finite temporal channel maps and positive ridge required')
    x=h-h.mean(-1,keepdim=True);cov=x@x.transpose(1,2)/h.shape[-1]
    var=cov.diagonal(dim1=1,dim2=2);regularized=var+ridge*var.mean(1,keepdim=True).clamp_min(1e-12)
    denominator=regularized.sqrt();return cov/(denominator[:,:,None]*denominator[:,None,:])


def consistency(raw,perturbed,mask,*,receivers=None,source_rxs=None):
    if (raw.ndim!=3 or raw.shape!=perturbed.shape or raw.shape[1]!=raw.shape[2]
        or mask.shape!=raw.shape[1:] or mask.dtype!=torch.bool or mask.device!=raw.device
        or torch.tril(mask).any() or not mask.any() or not torch.isfinite(raw).all() or not torch.isfinite(perturbed).all()):
        raise ValueError('Aligned finite correlations and nonempty strict upper mask required')
    # Per selected edge, then per packet: full/random/selective have equal units.
    loss=(raw-perturbed).square()[:,mask].mean(1)
    if receivers is None:return loss.mean()
    if (source_rxs is None or receivers.shape!=(len(loss),) or receivers.device!=loss.device
        or receivers.dtype!=torch.long or set(receivers.tolist())-set(source_rxs)):
        raise ValueError('Known source RX only')
    # Known RX labels balance observed groups; never interpreted as pure hardware.
    return torch.stack([loss[receivers==r].mean() for r in source_rxs if bool((receivers==r).any())]).mean()


class MaskStatistics:
    """Raw TX association conditional on RX/day, paired perturbation sensitivity."""
    def __init__(self,channels,*,source_rxs=(1,3,4,6,8),source_days=(1,2,3),device='cpu'):
        if channels<2 or len(set(source_rxs))!=5 or len(set(source_days))!=3:raise ValueError('Original fiveRX/threeday source required')
        self.channels=channels;self.rxs=tuple(source_rxs);self.days=tuple(source_days);self.device=device
        self.edges=torch.triu_indices(channels,channels,offset=1,device=device);e=self.edges.shape[1]
        self.count=torch.zeros(90,dtype=torch.float64,device=device)
        self.total=torch.zeros(90,e,dtype=torch.float64,device=device);self.square=torch.zeros_like(self.total)
        self.sensitivity=torch.zeros(5,e,dtype=torch.float64,device=device);self.rx_count=torch.zeros(5,dtype=torch.float64,device=device)
        self.samples=0;self.paired_views=0

    @torch.no_grad()
    def update(self,raw,perturbed,tx,rx,day,*,role):
        if (role!='L_s' or raw.ndim!=3 or raw.shape[1:]!=(self.channels,self.channels)
            or perturbed.ndim!=4 or perturbed.shape[1:]!=raw.shape or perturbed.shape[0]<1
            or any(v.shape!=(len(raw),) or v.dtype!=torch.long for v in (tx,rx,day))
            or bool(((tx<0)|(tx>=6)).any()) or set(rx.tolist())-set(self.rxs) or set(day.tolist())-set(self.days)
            or not torch.isfinite(raw).all() or not torch.isfinite(perturbed).all()):
            raise ValueError('Mask fitting requires true labeled source training data only')
        p=raw.detach().to(self.device,dtype=torch.float64);a=perturbed.detach().to(self.device,dtype=torch.float64)
        y=tx.to(self.device);r=torch.tensor([self.rxs.index(int(v)) for v in rx.tolist()],device=self.device)
        d=torch.tensor([self.days.index(int(v)) for v in day.tolist()],device=self.device)
        group=y*15+r*3+d;v=p[:,self.edges[0],self.edges[1]]
        sensitivity=(a-p[None]).square().mean(0)[:,self.edges[0],self.edges[1]]
        self.count.index_add_(0,group,torch.ones(len(v),dtype=torch.float64,device=self.device))
        self.total.index_add_(0,group,v);self.square.index_add_(0,group,v.square())
        self.sensitivity.index_add_(0,r,sensitivity)
        self.rx_count.index_add_(0,r,torch.ones(len(v),dtype=torch.float64,device=self.device))
        self.samples+=len(v);self.paired_views+=len(v)*len(a)

    def masks(self,*,fraction=.1,random_seed=2026102623,support_limited=False):
        if (not 0<fraction<=.25 or bool((self.count<=0).any()) or bool((self.rx_count<=0).any())):
            raise ValueError('All TX/RX/day cells and fixed sparse fraction required')
        mean=(self.total/self.count[:,None]).reshape(6,5,3,-1)
        within=(self.square/self.count[:,None]-(self.total/self.count[:,None]).square()).clamp_min(0).reshape(6,5,3,-1).mean((0,1,2))
        between=mean.var(0,unbiased=False).mean((0,1))
        identity=between/(between+within+1e-12)
        # A large median requires sensitivity in at least three of the five RX.
        by_rx=self.sensitivity/self.rx_count[:,None];sensitive=by_rx.median(0).values
        eligible=(identity<=torch.quantile(identity,.5))&(sensitive>=torch.quantile(sensitive,.75))&(sensitive>1e-12)
        count=max(1,int(self.edges.shape[1]*fraction));pool=eligible.nonzero().flatten()
        basic=dict(samples=self.samples,paired_views=self.paired_views,channels=self.channels,all_edges=self.edges.shape[1],
            requested_edges=count,eligible_edges=len(pool),fraction=fraction,random_seed=random_seed,
            selection_mode='natural_support_capped' if support_limited else 'fixed_fraction',
            source_cell_min=int(self.count.min()),source_cell_max=int(self.count.max()),
            identity_association_all_mean=float(identity.mean()),sensitivity_all_mean=float(sensitive.mean()))
        if not len(pool) or (len(pool)<count and not support_limited):raise InsufficientMaskEvidence(basic)
        if support_limited:count=min(count,len(pool))
        score=sensitive/(identity+1e-6)
        chosen=pool[torch.argsort(score[pool],descending=True,stable=True)[:count]]
        rng=torch.Generator(device=self.device).manual_seed(random_seed)
        random=torch.randperm(self.edges.shape[1],device=self.device,generator=rng)[:count]
        masks={}
        for name,indices in [('selective',chosen),('random',random),('full',torch.arange(self.edges.shape[1],device=self.device))]:
            mask=torch.zeros(self.channels,self.channels,dtype=torch.bool,device=self.device)
            mask[self.edges[0,indices],self.edges[1,indices]]=True;masks[name]=mask
        diagnostic=dict(**basic,selected_edges=count,
            identity_association_selected_mean=float(identity[chosen].mean()),
            sensitivity_selected_mean=float(sensitive[chosen].mean()),
            identity_association_random_mean=float(identity[random].mean()),sensitivity_random_mean=float(sensitive[random].mean()),
            random_selective_overlap=int(torch.isin(random,chosen).sum()),
            interpretation='conditional univariate TX association is not proof that selected correlations contain no identity information',
            selection_role='L_s_only',U_TX_labels_used=False,V_used=False,target_used=False)
        return masks,diagnostic
