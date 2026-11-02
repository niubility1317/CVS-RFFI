"""Small per-record complex FIR response for paired source-domain augmentation."""
import torch
from torch.nn import functional as F

def receiver_filter(x,coefficients):
 if x.ndim!=3 or x.shape[1]!=2 or coefficients.shape!=(len(x),4) or x.dtype!=coefficients.dtype or x.device!=coefficients.device or not torch.isfinite(x).all() or not torch.isfinite(coefficients).all():raise ValueError('Finite [batch,2,time] IQ and [batch,4] complex taps required')
 left=F.pad(x[:,:,:-1],(1,0));right=F.pad(x[:,:,1:],(0,1));ar,ai,br,bi=[coefficients[:,j,None] for j in range(4)];real=x[:,0]+ar*left[:,0]-ai*left[:,1]+br*right[:,0]-bi*right[:,1];imag=x[:,1]+ar*left[:,1]+ai*left[:,0]+br*right[:,1]+bi*right[:,0];scale=(1+coefficients.square().sum(1)).sqrt()[:,None,None];return torch.stack((real,imag),dim=1)/scale
