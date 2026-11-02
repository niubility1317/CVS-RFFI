"""Source-independent regularized blind-channel update for IQ restoration."""
import torch

def design_matrix(signal):
 if signal.ndim!=3 or signal.shape[1:]!=(2,256):raise ValueError('Expected [B,2,256] IQ')
 index=(torch.arange(256,device=signal.device)[:,None]-torch.arange(64,device=signal.device)[None,:]).abs();real=signal[:,0,index];imag=signal[:,1,index]
 return torch.cat((torch.cat((real,-imag),2),torch.cat((imag,real),2)),1)

def update_channel(signal,received,initial,regularization):
 if received.shape!=signal.shape or initial.shape!=(len(signal),2,64):raise ValueError('Expected paired IQ and initial complex kernel')
 if not isinstance(regularization,(int,float)) or not regularization>0:raise ValueError('Positive regularization required')
 X=design_matrix(signal);Xt=X.transpose(1,2);gram=torch.bmm(Xt,X);scale=gram.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-8);penalty=scale*regularization;eye=torch.eye(128,device=signal.device,dtype=signal.dtype).expand(len(signal),-1,-1);rhs=torch.bmm(Xt,received.reshape(len(signal),512,1)).squeeze(-1)+penalty[:,None]*initial.reshape(len(signal),128);solution=torch.linalg.solve(gram+penalty[:,None,None]*eye,rhs.unsqueeze(-1)).squeeze(-1);return solution.reshape_as(initial)


def write_fixed_predictions(path,indices,values,kernels):
 import hashlib
 import numpy as np
 v=np.from_dlpack(values.detach().cpu().contiguous());k=np.from_dlpack(kernels.detach().cpu().contiguous());idx=np.asarray(indices,dtype=np.int64)
 with path.open('xb') as f:np.savez_compressed(f,indices=idx,values=v,kernels=k)
 return v,k,hashlib.sha256(path.read_bytes()).hexdigest()
