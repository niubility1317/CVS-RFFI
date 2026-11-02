"""Compose a fixed real IQ action with a bounded complex FIR correction."""
import torch

def corrected_action(action,gain):
 if action.ndim!=3 or action.shape[1:]!=(512,512) or gain.shape!=(len(action),5) or not gain.is_complex():raise ValueError('Expected real action [N,512,512] and complex gain [N,5]')
 base=action.reshape(len(action),2,256,512);out=base.clone()
 for i,lag in enumerate((-2,-1,0,1,2)):
  shifted=torch.roll(base,shifts=lag,dims=2).clone()
  if lag>0:shifted[:,:,:lag,:]=0
  if lag<0:shifted[:,:,lag:,:]=0
  real=gain[:,i].real[:,None,None];imag=gain[:,i].imag[:,None,None];out[:,0]+=real*shifted[:,0]-imag*shifted[:,1];out[:,1]+=real*shifted[:,1]+imag*shifted[:,0]
 corrected=out.reshape(len(action),512,512)
 if not torch.isfinite(corrected).all():raise ValueError('Finite corrected action required')
 return corrected
