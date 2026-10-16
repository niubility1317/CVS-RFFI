"""Bounded same-label observation interpolation; no model or learned parameter."""
import math
import torch
@torch.no_grad()
def coherent_mix(x,donor):
 if x.shape!=donor.shape or x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all() and torch.isfinite(donor).all()):raise ValueError('Finite paired source IQ256 required')
 if x.dtype!=donor.dtype or x.device!=donor.device or x.dtype not in (torch.float32,torch.float64):raise ValueError('Matching original IQ dtype/device required')
 a=x.detach().double();b=donor.detach().double();ar,ai=a[:,0,80:160],a[:,1,80:160];br,bi=b[:,0,80:160],b[:,1,80:160]
 den=(br.square()+bi.square()).sum(1);power=(ar.square()+ai.square()).sum(1)
 real=(ar*br+ai*bi).sum(1)/(den+1e-12);imag=(ai*br-ar*bi).sum(1)/(den+1e-12)
 aligned=torch.stack((real[:,None]*b[:,0]-imag[:,None]*b[:,1],imag[:,None]*b[:,0]+real[:,None]*b[:,1]),1)
 proposed=.75*a+.25*aligned;norm=a.flatten(1).norm(dim=1);pnorm=proposed.flatten(1).norm(dim=1)
 if bool((norm<=1e-12).any()):raise ValueError('Nonzero source observations required')
 e=a/norm[:,None,None];u=proposed/pnorm.clamp_min(1e-12)[:,None,None];c=(e*u).flatten(1).sum(1).clamp(-1.,1.);h=u-c[:,None,None]*e;hn=h.flatten(1).norm(dim=1)
 theta=torch.atan2(hn,c);maximum=2*math.asin(.10/2);angle=theta.clamp_max(maximum)
 out=norm[:,None,None]*(angle.cos()[:,None,None]*e+angle.sin()[:,None,None]*h/hn.clamp_min(1e-12)[:,None,None])
 fallback=(pnorm<=1e-12)|(hn<=1e-12);out=torch.where(fallback[:,None,None],a,out).to(x.dtype)
 delta=(out.double()-a).flatten(1).norm(dim=1)/norm;energy=out.double().flatten(1).norm(dim=1)/norm;coherence=((real.square()+imag.square())*den.square())/((power*den).clamp_min(1e-24))
 if not bool(torch.isfinite(out).all()) or bool((delta>.10001).any()) or bool((energy-1).abs().gt(1e-5).any()):raise ValueError('Declared anchored IQ bound/norm violated')
 records=[dict(relative_L2=float(delta[i]),norm_ratio=float(energy[i]),reference_coherence=float(coherence[i].clamp(0,1)),scalar_real=float(real[i]),scalar_imag=float(imag[i]),arc_capped=bool(theta[i]>maximum),fallback=bool(fallback[i])) for i in range(len(x))]
 return out,records
