"""Batch real action matrices for a fixed complex 64-tap reflected filter."""
import torch

def action_matrix(kernel):
 if kernel.ndim!=3 or kernel.shape[1:]!=(2,64):raise ValueError('Expected [B,2,64] complex filter')
 n=len(kernel);index=(torch.arange(256,device=kernel.device)[:,None]-torch.arange(64,device=kernel.device)[None,:]).abs();index=index[None,:,:].expand(n,-1,-1);blocks=[]
 for c in (0,1):
  z=kernel.new_zeros((n,256,256));z.scatter_add_(2,index,kernel[:,c,None,:].expand(-1,256,-1));blocks.append(z)
 re,im=blocks;return torch.cat((torch.cat((re,-im),dim=2),torch.cat((im,re),dim=2)),dim=1)
