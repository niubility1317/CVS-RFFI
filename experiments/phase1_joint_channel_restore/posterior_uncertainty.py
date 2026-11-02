"""Gaussian-mixture posterior uncertainty for regularized blind channel updates."""
import math
import torch
from channel_update import design_matrix

@torch.no_grad()
def posterior_moments(y,action,noise,means,covariances,priors):
 if y.ndim!=2 or action.shape!=(len(y),y.shape[1],y.shape[1]) or noise.shape!=action.shape or means.ndim!=2 or means.shape[1]!=y.shape[1] or covariances.shape!=(len(means),y.shape[1],y.shape[1]) or priors.shape!=(len(means),):raise ValueError('Incompatible posterior dimensions')
 if not all(torch.isfinite(v).all() for v in (y,action,noise,means,covariances,priors)) or (priors<=0).any():raise ValueError('Finite positive mixture inputs required')
 cross=torch.matmul(covariances.unsqueeze(0),action.transpose(1,2).unsqueeze(1));s=torch.matmul(action.unsqueeze(1),cross)+noise.unsqueeze(1);s=(s+s.transpose(-1,-2))*.5;chol,info=torch.linalg.cholesky_ex(s)
 if (info!=0).any():raise ValueError('Positive definite observation covariance required')
 residual=y.unsqueeze(1)-torch.einsum('bij,kj->bki',action,means);sol=torch.cholesky_solve(residual.unsqueeze(-1),chol).squeeze(-1);mu=means.unsqueeze(0)+torch.matmul(cross,sol.unsqueeze(-1)).squeeze(-1);logdet=2*torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1);logw=torch.log(priors/priors.sum()).unsqueeze(0)-.5*((residual*sol).sum(-1)+logdet+y.shape[1]*math.log(2*math.pi));w=torch.softmax(logw,dim=1)
 temp=torch.linalg.solve_triangular(chol,cross.transpose(-1,-2),upper=False);components=covariances.unsqueeze(0)-torch.matmul(temp.transpose(-1,-2),temp);mean=(mu*w.unsqueeze(-1)).sum(1);diff=mu-mean[:,None,:];cov=(w[:,:,None,None]*(components+diff.unsqueeze(-1)*diff.unsqueeze(-2))).sum(1);cov=(cov+cov.transpose(-1,-2))*.5
 if not torch.isfinite(cov).all():raise ValueError('Finite posterior covariance required')
 return mean,cov

def uncertainty_gram(cov):
 if cov.ndim!=3 or cov.shape[1:]!=(512,512):raise ValueError('Expected [B,512,512] covariance')
 idx=(torch.arange(256,device=cov.device)[:,None]-torch.arange(64,device=cov.device)[None,:]).abs();i=idx[:,:,None];j=idx[:,None,:]
 rr=cov[:,:256,:256][:,i,j].sum(1);ii=cov[:,256:,256:][:,i,j].sum(1);ri=cov[:,:256,256:][:,i,j].sum(1);ir=cov[:,256:,:256][:,i,j].sum(1);diag=rr+ii;cross=ir-ri
 return torch.cat((torch.cat((diag,cross),2),torch.cat((cross.transpose(1,2),diag),2)),1)

def expected_channel_update(mean,cov,received,initial,regularization):
 if mean.shape!=received.shape or initial.shape!=(len(mean),2,64):raise ValueError('Expected paired IQ and initial complex kernel')
 if not isinstance(regularization,(int,float)) or not regularization>0:raise ValueError('Positive regularization required')
 X=design_matrix(mean);Xt=X.transpose(1,2);gram=torch.bmm(Xt,X);extra=uncertainty_gram(cov);scale=gram.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-8);penalty=scale*regularization;eye=torch.eye(128,device=mean.device,dtype=mean.dtype).expand(len(mean),-1,-1);rhs=torch.bmm(Xt,received.reshape(len(mean),512,1)).squeeze(-1)+penalty[:,None]*initial.reshape(len(mean),128);normal=gram+extra+penalty[:,None,None]*eye;return torch.linalg.solve(normal,rhs.unsqueeze(-1)).squeeze(-1).reshape_as(initial)
