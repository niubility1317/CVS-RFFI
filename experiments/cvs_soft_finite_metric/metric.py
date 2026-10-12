import torch
from torch.nn import functional as F
def fit(z,y,deltas,shrinkage=.1,eps=1e-6,hard_fraction=.8):
 z=z.double();deltas=deltas.double()
 if z.ndim!=2 or z.shape[1]!=160 or deltas.shape!=(4,len(z),160) or set(y.tolist())!=set(range(6)):raise ValueError('Expected complete six-class source features and four within-packet deltas')
 means=torch.stack([z[y==k].mean(0) for k in range(6)]);r=z-means[y]
 c=(r.T@r)/len(r);scale=c.trace()/160
 c=(1-shrinkage)*c+shrinkage*scale*torch.eye(160,dtype=z.dtype,device=z.device)+eps*scale.clamp_min(1e-12)*torch.eye(160,dtype=z.dtype,device=z.device)
 q=deltas.flatten(0,1);n=q.T@q/len(q)
 e,u=torch.linalg.eigh(c)
 if not (e>0).all():raise ValueError('Noise covariance not positive definite')
 whitening=(u*e.rsqrt()[None,:])@u.T
 effect=whitening@n@whitening;v,b=torch.linalg.eigh((effect+effect.T)/2);v=v.clamp_min(0);order=torch.arange(159,-1,-1,device=z.device);v=v[order];b=b[:,order]
 if float(v.sum())>1e-12:
  rank=int(torch.searchsorted(v.cumsum(0),hard_fraction*v.sum()).item())+1
 else:rank=0
 p=b[:,:rank]@b[:,:rank].T
 eye=torch.eye(160,dtype=z.dtype,device=z.device)
 return dict(means=means,MAHALANOBIS=torch.linalg.solve(c,eye),SOFT_FINITE=torch.linalg.solve(c+n,eye),HARD_FINITE=whitening@(eye-p)@whitening,finite_trace=float(n.trace()),noise_trace=float(c.trace()),hard_rank=rank,whitened_finite_eigenvalues=v,minimum_noise_eigenvalue=float(e.min()))
def logits(z,state,arm):
 z=z.double();mu=state['means']
 if arm=='COSINE_PROTO':return F.normalize(z,dim=1)@F.normalize(mu,dim=1).T
 if arm not in ('MAHALANOBIS','SOFT_FINITE','HARD_FINITE'):raise ValueError('Unknown fixed metric readout')
 residual=z[:,None,:]-mu[None,:,:];m=state[arm]
 return -.5*torch.einsum('bci,ij,bcj->bc',residual,m,residual)
