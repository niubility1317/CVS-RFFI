"""Frozen selected self-reference restoration, input-only."""
import torch

@torch.no_grad()
def apply(model,phase,gate,alpha,threshold,x):
 enabled=gate(x).sigmoid()>threshold;out=x.clone()
 if enabled.any():
  a=phase.anchor(x[enabled]);out[enabled]=a+alpha*(model(a)-a)
 return out,enabled
