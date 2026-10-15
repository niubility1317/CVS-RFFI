"""Project a displacement onto visible-source identity-risk halfspaces."""
import itertools
import numpy as np
import torch

def project(increment,risk_gradients):
    if (increment.ndim!=1 or risk_gradients.ndim!=2 or risk_gradients.shape[1]!=increment.numel()
            or not 1<=risk_gradients.shape[0]<=5 or not torch.isfinite(increment).all()
            or not torch.isfinite(risk_gradients).all()):raise ValueError('Finite feature displacement and one to five source risk gradients required')
    gradients=risk_gradients.double();norms=gradients.norm(dim=1)
    if bool((norms<=0).any()):raise ValueError('Each source RX risk must have a nonzero gradient')
    rows=gradients/norms[:,None];u=increment.double();length=float(u.norm())
    if not length:return dict(projected=u.clone(),retained_norm_ratio=0.,cosine_raw=None,active=[],normalized_max_violation=0.)
    v=u/length;gram=(rows@rows.T).cpu().numpy();b=(rows@v).cpu().numpy();candidates=[];tol=1e-8
    for size in range(len(b)+1):
        for subset in itertools.combinations(range(len(b)),size):
            coefficients=np.zeros(len(b),dtype=np.float64)
            if subset:
                active=list(subset);solution=np.linalg.lstsq(gram[np.ix_(active,active)],b[active],rcond=1e-12)[0]
                if np.any(solution < -tol):continue
                coefficients[active]=np.maximum(solution,0.)
            residual=b-gram@coefficients
            if np.max(residual)>tol or np.max(np.abs(coefficients*residual))>tol:continue
            objective=float(coefficients@gram@coefficients)
            candidates.append((objective,tuple(subset),coefficients))
    if not candidates:raise ValueError('Source-risk projection has no numerically valid KKT solution')
    _,active,coefficients=min(candidates,key=lambda item:item[0])
    p=(v-rows.T@torch.as_tensor(coefficients,device=u.device))*length
    violation=float((rows@p).max())/length
    if violation>1e-7 or float(p.norm())>length*(1+1e-7):raise ValueError('Projection violated its source-risk cone/norm')
    retained=float(p.norm())/length
    return dict(projected=p,retained_norm_ratio=retained,
        cosine_raw=float(torch.dot(p,u)/(p.norm()*u.norm())) if retained>1e-12 else None,
        active=list(active),normalized_max_violation=violation)

def match_stored_norm(base,raw,target):
    """Match the norm after storing candidate positions in original FP32."""
    if base.dtype!=torch.float32 or base.shape!=raw.shape or base.shape!=target.shape:
        raise ValueError('Equal-shape FP32 feature positions required')
    if not all(bool(torch.isfinite(v).all()) for v in (base,raw,target)):raise ValueError('Finite feature positions required')
    u=raw.double()-base.double();wanted=float((target.double()-base.double()).norm())
    if not wanted:return base.clone(),dict(scale=0.,relative_norm_error=0.,norm_matched=True)
    def position(factor):return (base.double()+factor*u).float()
    def norm(value):return float((value.double()-base.double()).norm())
    if not float(u.norm()):raise ValueError('Nonzero target cannot be matched by zero original displacement')
    lo,hi=0.,1.
    while norm(position(hi))<wanted:
        hi*=2
        if hi>16:raise ValueError('Stored norm control differs materially from the raw displacement')
    best=(float('inf'),None,None)
    for _ in range(64):
        factor=(lo+hi)/2;value=position(factor);actual=norm(value);error=abs(actual-wanted)
        if error<best[0]:best=(error,factor,value)
        if actual<wanted:lo=factor
        else:hi=factor
    error,factor,value=best
    return value,dict(scale=factor,relative_norm_error=error/wanted,norm_matched=error/wanted<=1e-3)
