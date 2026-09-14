"""Explicit signed fields, with no higher derivatives through GRL."""
import torch
import torch.nn.functional as F
from torch.func import functional_call

def tx_partition(y,domains,step):
    tx=y.unique(sorted=True).roll(step%len(y.unique()))
    if len(tx)<2:raise ValueError('XT needs at least two identities')
    a=torch.isin(y,tx[:len(tx)//2]);b=~a
    if not torch.equal(domains[a].unique(sorted=True),domains.unique(sorted=True)) or not torch.equal(domains[b].unique(sorted=True),domains.unique(sorted=True)):
        raise ValueError('both TX sets must cover the full source domain universe')
    return a,b

def cross_tx_fields(head,z,y,domain,step,lr,mu,lambda_encoder,exposure_control=False):
    a,b=tx_partition(y,domain,step)
    params=dict(head.named_parameters());names=list(params);values=tuple(params.values())
    def ce(features,mask,p):return F.cross_entropy(functional_call(head,p,(features[mask],)),domain[mask])
    if exposure_control:
        loss=mu*F.cross_entropy(head(z.detach()),domain)
        return loss,dict(exposure_control=True,episode_samples=len(y),encoder_adv_norm_scope='none')
    temporary=[];diagnostic={}
    for label,fit,monitor in [('A_to_B',a,b),('B_to_A',b,a)]:
        inner=ce(z.detach(),fit,params)
        gradients=torch.autograd.grad(inner,values,create_graph=True)
        adapted={n:p-lr*g for n,p,g in zip(names,values,gradients)}
        temporary.append((adapted,monitor))
        diagnostic[label]=dict(fit_change=float((ce(z.detach(),fit,adapted)-inner).detach()),
            monitor_change=float((ce(z.detach(),monitor,adapted)-ce(z.detach(),monitor,params)).detach()))
    meta=sum(ce(z.detach(),mask,p) for p,mask in temporary)*mu/2
    adversarial=-lambda_encoder/2*sum(ce(z,mask,{n:p.detach() for n,p in ps.items()}) for ps,mask in temporary)
    diagnostic.update(episode_samples=len(y),tx_A=y[a].unique().tolist(),tx_B=y[b].unique().tolist(),
        head_inner_commits=0,inner_steps=1,meta_second_order=True,encoder_input_gradient=True)
    return meta+adversarial,diagnostic

def procrustes(z0,zp,rho=.01):
    # Row-major samples of the actual head input; no feature normalization.
    if z0.shape!=zp.shape or z0.ndim!=2:raise ValueError('paired actual head input required')
    d=z0.shape[1];eye=torch.eye(d,device=z0.device,dtype=z0.dtype)
    covariance=zp.T@z0/len(z0)
    regularizer=rho*(z0.square().sum(1).mean()/d).clamp_min(1e-12)
    u,_,vh=torch.linalg.svd(covariance+regularizer*eye)
    signs=torch.ones(d,device=z0.device,dtype=z0.dtype);signs[-1]=torch.linalg.det(u@vh)
    return ((u*signs)@vh).detach()
