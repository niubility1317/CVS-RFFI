"""Exact per-observation derivatives of the existing cosine W and gate convolutions."""
import torch
from torch.nn import functional as F

def cosine(a,b):
 if a.shape!=b.shape or a.ndim!=2 or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Finite paired gradient matrices required')
 a=a.detach().double();b=b.detach().double();na=a.norm(dim=1);nb=b.norm(dim=1);valid=(na>1e-12)&(nb>1e-12)
 v=(a*b).sum(1)/(na*nb).clamp_min(1e-300)
 return torch.where(valid,v.clamp(-1,1),torch.full_like(v,float('nan'))),valid

OBJECTIVES=('margin','CE0','CE01')

def head_weight_vectors(joint,logits,y,weight,objective):
 n=len(joint)
 if joint.shape!=(n,160) or logits.shape!=(n,6) or weight.shape!=(6,160) or y.shape!=(n,) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not all(torch.isfinite(t).all() for t in (joint,logits,weight)):raise ValueError('Actual finite defaultcos30/smoothed six-class inputs required')
 with torch.no_grad():
  x=F.normalize(joint.detach(),dim=1,eps=1e-4);w=weight.detach();norm=w.norm(dim=1);den=norm.clamp_min(1e-4);u=w/den[:,None]
  if objective not in OBJECTIVES:raise ValueError('Fixed diagnostic objective required')
  target=F.one_hot(y,6).to(logits)
  if objective=='margin':
   competitors=logits.detach().clone();competitors[torch.arange(n,device=y.device),y]=-torch.inf;c=competitors.argmax(1);delta=(F.one_hot(c,6).to(logits)-target)/30
  else:
   smoothing=0. if objective=='CE0' else .01;delta=logits.detach().softmax(1)-(target*(1-smoothing)+smoothing/6)
  dot=x@u.T;tangent=(x[:,None,:]-dot[:,:,None]*u[None,:,:])/den[None,:,None]
  tangent=torch.where((norm>=1e-4)[None,:,None],tangent,x[:,None,:]/1e-4)
  return (30*delta[:,:,None]*tangent).flatten(1)

def conv_vectors(inputs,output_grad,module):
 k=module.kernel_size[0]
 if module.groups!=1 or module.stride!=(1,) or module.dilation!=(1,) or module.padding!=(k//2,) or module.bias is None or k%2!=1 or inputs.ndim!=3 or output_grad.ndim!=3 or inputs.shape[0]!=output_grad.shape[0] or inputs.shape[2]!=output_grad.shape[2] or inputs.shape[1]!=module.in_channels or output_grad.shape[1]!=module.out_channels or not torch.isfinite(inputs).all() or not torch.isfinite(output_grad).all():raise ValueError('Actual stride1 odd-kernel padded gate convolution required')
 with torch.no_grad():
  patches=F.pad(inputs.detach(),(k//2,k//2)).unfold(-1,k,1);g=output_grad.detach();dw=torch.einsum('bot,bitk->boik',g,patches);db=g.sum(-1)
  return torch.cat((dw.flatten(1),db),1)
