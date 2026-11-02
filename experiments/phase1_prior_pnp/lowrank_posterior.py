"""Low-rank Gaussian observation posterior, independent of DURA."""
import math
import torch

def factors_from_covariance(covariance,rank):
 if covariance.ndim!=3 or covariance.shape[-1]!=covariance.shape[-2] or not 1<=rank<=covariance.shape[-1] or not torch.isfinite(covariance).all():raise ValueError('Finite square covariance and valid rank required')
 symmetric=(covariance+covariance.transpose(-1,-2))*.5;values,vectors=torch.linalg.eigh(symmetric);tol=torch.finfo(values.dtype).eps*covariance.shape[-1]*values.abs().amax().clamp_min(1.)*10
 if (values < -tol).any():raise ValueError('Positive semidefinite covariance required')
 return vectors[:,:,-rank:]*values[:,-rank:].clamp_min(0).sqrt().unsqueeze(-2)

def lowrank_posterior(y,action,variance,means,factors,priors):
 if y.ndim!=2:raise ValueError('Batched observation required')
 batch,dim=y.shape;classes=means.shape[0];rank=factors.shape[-1]
 if action.shape!=(batch,dim,dim) or variance.shape!=(batch,) or means.shape!=(classes,dim) or factors.shape!=(classes,dim,rank) or priors.shape!=(classes,) or rank<1 or not all(torch.isfinite(v).all() for v in (y,action,variance,means,factors,priors)) or (variance<=0).any() or (priors<=0).any():raise ValueError('Compatible finite inputs and positive variance and priors required')
 B=action[:,None]@factors[None];residual=y[:,None]-torch.einsum('bij,kj->bki',action,means);noise=variance[:,None,None,None];eye=torch.eye(rank,dtype=y.dtype,device=y.device);gram=eye+B.transpose(-1,-2)@B/noise;chol,info=torch.linalg.cholesky_ex(gram)
 if (info!=0).any():raise ValueError('Positive definite low-rank system required')
 rhs=(B.transpose(-1,-2)@residual.unsqueeze(-1))/noise;weights=torch.cholesky_solve(rhs,chol);solve=(residual.unsqueeze(-1)-B@weights)/noise;states=means[None]+(factors[None]@(B.transpose(-1,-2)@solve)).squeeze(-1)
 logdet=dim*torch.log(variance[:,None])+2*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1);logweight=torch.log(priors/priors.sum())[None]-.5*((residual*solve.squeeze(-1)).sum(-1)+logdet+dim*math.log(2*math.pi))
 if not torch.isfinite(states).all() or not torch.isfinite(logweight).all():raise ValueError('Finite posterior required')
 return states,logweight-torch.logsumexp(logweight,dim=1,keepdim=True)
