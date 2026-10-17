"""Bounded proper-complex unitary observation on the existing finite IQ window.

T is symmetric nearest-neighbour averaging, with half-sample symmetric ends.
L=(T-trace(T)/N I)/rho has spectrum in [-1,1].  exp(i alpha L)
preserves energy; its eighth-order Taylor implementation has <=3.1e-15
operator error at |alpha|<=.1 in exact arithmetic. No packet wrap or new IQ.
"""
import math
import torch
LENGTH=256;BOUND=.1;ORDER=8
RHO=math.cos(math.pi/LENGTH)+1./LENGTH

def generator(x):
 left=torch.cat((x[...,:1],x[...,:-1]),-1)
 right=torch.cat((x[...,1:],x[...,-1:]),-1)
 return (.5*(left+right)-x/LENGTH)/RHO

def apply(x,alpha):
 if not torch.is_tensor(x) or x.ndim!=3 or x.shape[1:]!=(2,LENGTH) or x.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(x).all()):raise ValueError('Finite N,2,256 float IQ only')
 if not torch.is_tensor(alpha):alpha=torch.full((len(x),),float(alpha),dtype=x.dtype,device=x.device)
 if alpha.shape!=(len(x),) or alpha.dtype!=x.dtype or alpha.device!=x.device or not bool(torch.isfinite(alpha).all()) or bool((alpha.abs()>BOUND).any()):raise ValueError('Declared per-packet phase bound only')
 value=x.clone();term=x;a=alpha[:,None]
 for k in range(1,ORDER+1):
  t=generator(term);term=torch.stack((-a*t[:,1]/k,a*t[:,0]/k),1);value=value+term
 return value

def profile(dtype=torch.float64,device='cpu'):
 k=torch.arange(LENGTH,dtype=dtype,device=device)
 return (torch.cos(math.pi*k/LENGTH)-1./LENGTH)/RHO

def error_bound():return math.exp(BOUND)*BOUND**(ORDER+1)/math.factorial(ORDER+1)
