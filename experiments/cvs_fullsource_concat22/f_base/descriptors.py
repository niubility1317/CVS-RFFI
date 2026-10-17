"""Descriptor formulas reused from pinned sequence50 commit990ffabe0; no data access."""
import torch
ARMS=('MEAN','MEAN_VAR','NONLINEAR_MEAN','RFF','COV','SIG','SIG_TIME')
def distribution_descriptor(u,mode,omega):
 if u.ndim!=3 or u.shape[-1]!=6 or omega.shape!=(14,6) or mode not in ARMS[:4]:raise ValueError('Fixed local distribution dimensions/mode differ')
 a=u@omega.T;mu=a.mean(1)
 if mode=='MEAN':return torch.cat((mu,mu),-1)
 if mode=='MEAN_VAR':return torch.cat((mu,a.var(1,unbiased=False)),-1)
 if mode=='NONLINEAR_MEAN':return torch.cat((mu.cos(),mu.sin()),-1)
 return torch.cat((a.cos().mean(1),a.sin().mean(1)),-1)

def path_signature(u,include_time):
 if u.ndim!=3 or u.shape[1]<2 or u.shape[-1]!=6 or not isinstance(include_time,bool):raise ValueError('A six-coordinate path with multiple positions is required')
 if include_time:
  t=torch.linspace(0,1,u.shape[1],device=u.device,dtype=u.dtype)[None,:,None].expand(len(u),-1,-1)
  u=torch.cat((u,t),-1)
 delta=u[:,1:]-u[:,:-1];before=delta.cumsum(1)-delta
 second=.5*((before[:,:,:,None]*delta[:,:,None,:])-(delta[:,:,:,None]*before[:,:,None,:])).sum(1)
 ij=torch.triu_indices(u.shape[-1],u.shape[-1],offset=1,device=u.device)
 return torch.cat((delta.sum(1),second[:,ij[0],ij[1]]),-1)

def moment_descriptor(u):
 if u.ndim!=3 or u.shape[-1]!=6 or u.shape[1]<2:raise ValueError('A six-coordinate multi-position path is required')
 mu=u.mean(1);centered=u-mu[:,None];cov=centered.transpose(1,2)@centered/u.shape[1]
 ij=torch.triu_indices(6,6,device=u.device)
 return torch.cat((mu,cov[:,ij[0],ij[1]]),-1)
