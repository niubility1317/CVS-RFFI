"""Receiver CAGrad adaptation; default W is excluded from gradient correction."""
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn import functional as F
RX=(1,3,4,6,8)
CLASS_W='id_backbone.encoder.core.id_backbone.cls_head.weight'
def solve(gram,weights,c=.5):
 A=np.asarray(gram,dtype=np.float64);b=np.asarray(weights,dtype=np.float64)
 if c!=.5 or A.shape!=(len(b),len(b)) or not len(b) or not np.isfinite(A).all() or not np.isfinite(b).all() or (b<=0).any() or abs(b.sum()-1)>1e-6 or not np.allclose(A,A.T,rtol=1e-6,atol=1e-8):raise ValueError('Invalid fixed receiver gradient Gram')
 if np.linalg.eigvalsh(A).min() < -1e-8*max(float(np.abs(A).max()),1e-20):raise ValueError('NonPSD receiver Gram')
 n=float(b@A@b);fallback=False;reason=None
 if n<=1e-24:return dict(coefficients=b.tolist(),norm_control_scale=1.,fallback=False,reason='zero_reference',relative_gap=0.,norm_ratio=1.,min_reference_dot=float((A@b).min()),min_selected_dot=float((A@b).min()),direction_change=0.,trust_relative=0.)
 B=A/n;ab=B@b;r=c;eps=1e-20
 def obj(w):return float(w@ab+r*np.sqrt(max(float(w@B@w),0.)+eps))
 def jac(w):return ab+r*(B@w)/np.sqrt(max(float(w@B@w),0.)+eps)
 opt=minimize(obj,b.copy(),jac=jac,bounds=[(0.,1.)]*len(b),constraints=[dict(type='eq',fun=lambda w:w.sum()-1.,jac=lambda w:np.ones_like(w))],method='SLSQP',options=dict(maxiter=100,ftol=1e-10))
 w=np.clip(opt.x,0.,1.);w=w/w.sum() if w.sum()>0 else b.copy();gn=np.sqrt(max(float(w@B@w),0.));a=b+r*w/max(gn,1e-20);dots=B@a;gap=obj(w)-float(dots.min());trust=np.sqrt(max(float((a-b)@B@(a-b)),0.));ratio=np.sqrt(max(float(a@B@a),0.))
 if not opt.success or not np.isfinite(a).all() or not np.isfinite(gap) or gap>1e-4 or gap< -1e-6 or trust>c+1e-5 or ratio<=0:
  fallback=True;reason='solver_or_duality_check';a=b.copy();ratio=1.;gap=0.;trust=0.;dots=B@b
 return dict(coefficients=a.tolist(),norm_control_scale=ratio,fallback=fallback,reason=reason,relative_gap=float(gap),norm_ratio=float(ratio),min_reference_dot=float((A@b).min()),min_selected_dot=float((A@a).min()),direction_change=float(np.sqrt(max(float((a-ratio*b)@B@(a-ratio*b)),0.))),trust_relative=float(trust))
def correction(model,logits,y,rx,mode,*,role='L_s'):
 if role!='L_s' or mode not in ('NORM_CONTROL','RX_CAGRAD') or logits.shape!=(len(y),6) or y.ndim!=1 or y.dtype!=torch.long or rx.dtype!=torch.long or rx.shape!=y.shape or not len(y) or set(rx.tolist())-set(RX) or (y<0).any() or (y>=6).any() or not torch.isfinite(logits).all():raise ValueError('Legal native labelled receiver CE required')
 pairs=[(n,p) for n,p in model.named_parameters() if n.startswith('id_backbone.') and n!=CLASS_W]
 if sum(p.numel() for n,p in pairs)!=195041 or any(not p.requires_grad for n,p in pairs):raise ValueError('Actual fullfeature boundary or jointgradient disabled')
 params=[p for n,p in pairs];per=F.cross_entropy(logits,y,label_smoothing=.01,reduction='none');groups=[r for r in RX if (rx==r).any()];counts=torch.stack([(rx==r).sum() for r in groups]);losses=[per[rx==r].mean() for r in groups];vectors=[]
 for loss in losses:
  grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);vectors.append(torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1) for p,g in zip(params,grad)]))
 G=torch.stack(vectors);weights=counts.to(G)/counts.sum();g0=weights@G;gram=(G.double()@G.double().T).cpu().numpy();s=solve(gram,weights.double().cpu().numpy());a=torch.tensor(s['coefficients'],device=G.device,dtype=G.dtype);target=a@G if mode=='RX_CAGRAD' else g0*s['norm_control_scale'];delta=target-g0;linear=logits.sum()*0;offset=0;selected={}
 for n,p in pairs:
  dv=delta[offset:offset+p.numel()].reshape_as(p);linear=linear+((p-p.detach())*dv.detach()).sum();
  if n.endswith(('t1.pw.weight','t2.pw.weight','sequence_pool.views.0.local_projection.weight')):selected[n]=dict(original=g0[offset:offset+p.numel()].detach().cpu(),target=target[offset:offset+p.numel()].detach().cpu())
  offset+=p.numel()
 detail=dict(groups=groups,group_counts=counts.cpu().tolist(),group_mean_CE=[float(v.detach()) for v in losses],gram=gram.tolist(),weights=weights.cpu().tolist(),**s)
 return linear,detail,selected
