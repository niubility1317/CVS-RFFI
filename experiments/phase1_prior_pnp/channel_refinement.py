"""Bounded complex FIR observation correction for blind-operator diagnostics."""
import torch

def fir_design(signal):
 if signal.ndim!=2 or not signal.is_complex() or signal.shape[1]!=256:raise ValueError('Complex [N,256] signal required')
 cols=[]
 for lag in (-2,-1,0,1,2):
  x=torch.roll(signal,shifts=lag,dims=1).clone()
  if lag>0:x[:,:lag]=0
  if lag<0:x[:,lag:]=0
  cols.append(x)
 return torch.stack(cols,dim=2)

def fit_bounded_delta(design,target,train,alpha=.1,cap=.25):
 if design.ndim!=3 or design.shape[1:]!=(256,5) or target.shape!=design.shape[:2] or train.shape!=(256,) or train.dtype!=torch.bool or alpha<=0 or cap<=0:raise ValueError('Valid source FIR fit contract required')
 a=design[:,train,:];b=target[:,train];gram=a.conj().transpose(1,2)@a;rhs=(a.conj().transpose(1,2)@b.unsqueeze(-1)).squeeze(-1);scale=gram.diagonal(dim1=-2,dim2=-1).real.mean(1).clamp_min(1e-8);eye=torch.eye(5,device=design.device,dtype=design.dtype);g=torch.linalg.solve(gram+alpha*scale[:,None,None]*eye[None],rhs);norm=torch.linalg.vector_norm(g,dim=1).clamp_min(1e-12);return g*torch.clamp(cap/norm,max=1)[:,None]

def observation_metrics(received,true_forward,parent_forward):
 if received.shape!=true_forward.shape or received.shape!=parent_forward.shape or received.ndim!=2 or received.shape[1]!=256:raise ValueError('Matching complex source signals required')
 gt=fir_design(true_forward);gp=fir_design(parent_forward);even=(torch.arange(256,device=received.device)%2)==0;den=received.abs().square().sum(1).clamp_min(1e-8);rows=[]
 for train in (even,~even):
  test=~train;oracle=fit_bounded_delta(gt,received-true_forward,train);blind=fit_bounded_delta(gp,received-parent_forward,train);apply=lambda design,g:torch.bmm(design,g.unsqueeze(-1)).squeeze(-1);base=(received-true_forward)[:,test].abs().square().sum(1)/den;oracle_err=(received-true_forward-apply(gt,oracle))[:,test].abs().square().sum(1)/den;blind_err=(received-true_forward-apply(gt,blind))[:,test].abs().square().sum(1)/den;parent_err=(received-parent_forward)[:,test].abs().square().sum(1)/den;parent_fit=(received-parent_forward-apply(gp,blind))[:,test].abs().square().sum(1)/den;rows.append(torch.stack((base,oracle_err,blind_err,parent_err,parent_fit,torch.linalg.vector_norm(oracle,dim=1),torch.linalg.vector_norm(blind,dim=1)),1))
 out=rows[0]+rows[1];out[:,5:]/=2
 if not torch.isfinite(out).all():raise ValueError('Finite channel diagnosis required')
 return out
